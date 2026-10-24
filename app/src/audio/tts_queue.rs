use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

use anyhow::Result;

use crate::sidecar::{SidecarClient, TtsRequest, TtsRun};

use super::{Player, SentenceSplitter, speakable};

/// Turns one sentence into WAV bytes (the sidecar in the app, a fake in tests).
type Synthesize = Box<dyn Fn(&str) -> Result<Vec<u8>> + Send>;

/// Where synthesized audio goes (the [`Player`] in the app, a fake in tests).
trait Output: Send + Sync + 'static {
    fn enqueue(&self, wav: Vec<u8>);
    fn stop(&self);
    fn is_playing(&self) -> bool;
}

impl Output for Player {
    fn enqueue(&self, wav: Vec<u8>) {
        Player::enqueue(self, wav);
    }
    fn stop(&self) {
        Player::stop(self);
    }
    fn is_playing(&self) -> bool {
        Player::is_playing(self)
    }
}

struct Job {
    generation: u64,
    text: String,
}

/// State shared with the worker.
struct Shared {
    /// Bumped by `stop`; jobs and results of older generations are dropped.
    generation: AtomicU64,
    /// Sentences of the current generation not yet handed to the output.
    pending: AtomicUsize,
}

impl Shared {
    fn done_one(&self, generation: u64) {
        if self.generation.load(Ordering::SeqCst) == generation {
            let _ = self
                .pending
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
        }
    }
}

/// Speaks a streaming reply sentence by sentence: `push` text deltas as they
/// arrive; completed sentences are synthesized with `/tts` on a worker thread
/// strictly in order and enqueued on the [`Player`], so playback starts with
/// the first sentence while generation continues.
///
/// Each sentence is cleaned with [`speakable`] first (markdown, code, URLs and
/// emoji are not read). After a connection error the rest of the reply is not
/// synthesized; other errors skip just that sentence. Each distinct error is
/// reported once.
///
/// Dropping the queue does not stop audio already playing; call [`Self::stop`].
pub struct SpeechQueue {
    splitter: SentenceSplitter,
    jobs: mpsc::Sender<Job>,
    shared: Arc<Shared>,
    output: Arc<dyn Output>,
    errors: async_channel::Receiver<String>,
}

impl SpeechQueue {
    pub fn new(
        client: SidecarClient,
        player: Player,
        profile_id: String,
        run: TtsRun,
        speed: Option<f32>,
    ) -> Self {
        let synthesize: Synthesize = Box::new(move |text: &str| {
            client.tts(&TtsRequest {
                profile_id: profile_id.clone(),
                text: text.to_string(),
                engine: None,
                run: run.clone(),
                speed,
            })
        });
        Self::with_backend(synthesize, Arc::new(player))
    }

    fn with_backend(synthesize: Synthesize, output: Arc<dyn Output>) -> Self {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
        });
        let (jobs, rx) = mpsc::channel::<Job>();
        let (error_tx, errors) = async_channel::unbounded();
        let worker_shared = shared.clone();
        let worker_output = output.clone();
        let spawned = thread::Builder::new()
            .name("seda-tts-queue".into())
            .spawn(move || worker(rx, synthesize, worker_output, worker_shared, error_tx));
        if let Err(err) = spawned {
            // `push` then sends into a closed channel and nothing is spoken.
            eprintln!("seda: cannot start the speech worker: {err}");
        }
        Self {
            splitter: SentenceSplitter::new(),
            jobs,
            shared,
            output,
            errors,
        }
    }

    /// Feed a delta of the reply.
    pub fn push(&mut self, delta: &str) {
        for sentence in self.splitter.push(delta) {
            self.submit(&sentence);
        }
    }

    /// The reply is complete: speak the remainder.
    pub fn finish(&mut self) {
        if let Some(rest) = self.splitter.flush() {
            self.submit(&rest);
        }
    }

    fn submit(&self, sentence: &str) {
        let Some(text) = speakable(sentence) else {
            return;
        };
        let generation = self.shared.generation.load(Ordering::SeqCst);
        self.shared.pending.fetch_add(1, Ordering::SeqCst);
        if self.jobs.send(Job { generation, text }).is_err() {
            self.shared.done_one(generation);
        }
    }

    /// Cancel pending synthesis and stop playback.
    pub fn stop(&mut self) {
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        self.shared.pending.store(0, Ordering::SeqCst);
        self.splitter.reset();
        self.output.stop();
    }

    /// Errors from synthesis (e.g. sidecar offline), to show as notifications.
    /// Each error is reported once.
    pub fn errors(&self) -> async_channel::Receiver<String> {
        self.errors.clone()
    }

    /// Whether sentences are still being synthesized or played.
    pub fn is_active(&self) -> bool {
        self.shared.pending.load(Ordering::SeqCst) > 0 || self.output.is_playing()
    }
}

/// Synthesizes jobs in order. Enqueueing on the output does not block, so
/// sentence N+1 is synthesized while N plays.
fn worker(
    jobs: mpsc::Receiver<Job>,
    synthesize: Synthesize,
    output: Arc<dyn Output>,
    shared: Arc<Shared>,
    errors: async_channel::Sender<String>,
) {
    let mut reported = HashSet::new();
    // The generation in which the sidecar was unreachable: skip its other jobs.
    let mut unreachable: Option<u64> = None;
    let current = |generation| shared.generation.load(Ordering::SeqCst) == generation;

    while let Ok(job) = jobs.recv() {
        if !current(job.generation) || unreachable == Some(job.generation) {
            shared.done_one(job.generation);
            continue;
        }
        let result = synthesize(&job.text);
        // `stop` may have been called during synthesis.
        if current(job.generation) {
            match result {
                Ok(wav) => output.enqueue(wav),
                Err(err) => {
                    let message = format!("{err:#}");
                    if is_connection_error(&message) {
                        unreachable = Some(job.generation);
                    }
                    if reported.insert(message.clone()) {
                        let _ = errors.try_send(message);
                    }
                }
            }
        }
        shared.done_one(job.generation);
    }
}

/// `SidecarClient` reports transport failures (refused, timed out, DNS)
/// with this prefix, and HTTP status errors as `sidecar <code>: ...`.
fn is_connection_error(message: &str) -> bool {
    message.starts_with("سایدکار در دسترس نیست")
        || message.contains("Connection refused")
        || message.contains("connection refused")
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Mutex,
        time::{Duration, Instant},
    };

    use anyhow::anyhow;

    use super::*;

    #[derive(Default)]
    struct FakeOutput {
        played: Mutex<Vec<String>>,
        stops: AtomicUsize,
    }

    impl Output for FakeOutput {
        fn enqueue(&self, wav: Vec<u8>) {
            self.played
                .lock()
                .unwrap()
                .push(String::from_utf8(wav).unwrap());
        }
        fn stop(&self) {
            self.stops.fetch_add(1, Ordering::SeqCst);
        }
        fn is_playing(&self) -> bool {
            false
        }
    }

    fn wait_idle(queue: &SpeechQueue) {
        let start = Instant::now();
        while queue.is_active() {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "queue never went idle"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn fake_queue(
        synthesize: impl Fn(&str) -> Result<Vec<u8>> + Send + 'static,
    ) -> (SpeechQueue, Arc<FakeOutput>) {
        let output = Arc::new(FakeOutput::default());
        let queue = SpeechQueue::with_backend(Box::new(synthesize), output.clone());
        (queue, output)
    }

    #[test]
    fn speaks_in_order_with_variable_latency() {
        let (mut queue, output) = fake_queue(|text| {
            // Later sentences are faster: order must still hold.
            let ms = 40u64.saturating_sub(text.chars().count() as u64);
            thread::sleep(Duration::from_millis(ms));
            Ok(text.as_bytes().to_vec())
        });
        for delta in [
            "سلام **دوست** من! ",
            "حالت",
            " چطوره؟\n",
            "```\ncode\n```\n",
            "خوبم. ",
            "مرسی",
        ] {
            queue.push(delta);
        }
        assert!(queue.is_active());
        queue.finish();
        wait_idle(&queue);
        assert_eq!(
            *output.played.lock().unwrap(),
            vec!["سلام دوست من!", "حالت چطوره؟", "خوبم.", "مرسی"]
        );
    }

    #[test]
    fn stop_discards_in_flight_and_pending() {
        let (mut queue, output) = fake_queue(|text| {
            thread::sleep(Duration::from_millis(100));
            Ok(text.as_bytes().to_vec())
        });
        queue.push("جمله اول. جمله دوم. جمله سوم. ");
        thread::sleep(Duration::from_millis(30));
        queue.stop();
        assert!(!queue.is_active());
        assert_eq!(output.stops.load(Ordering::SeqCst), 1);
        thread::sleep(Duration::from_millis(400));
        assert!(output.played.lock().unwrap().is_empty());

        // The queue keeps working after a stop.
        queue.push("دوباره شروع کنیم. ");
        queue.finish();
        wait_idle(&queue);
        assert_eq!(*output.played.lock().unwrap(), vec!["دوباره شروع کنیم."]);
    }

    #[test]
    fn connection_error_reported_once_and_stops() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let (mut queue, output) = fake_queue(move |_| {
            c.fetch_add(1, Ordering::SeqCst);
            Err(anyhow!("سایدکار در دسترس نیست: connection refused"))
        });
        queue.push("یک جمله. دو جمله. سه جمله. ");
        queue.finish();
        wait_idle(&queue);
        let errors = queue.errors();
        assert!(errors.try_recv().unwrap().starts_with("سایدکار"));
        assert!(errors.try_recv().is_err(), "reported once");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "rest not synthesized");
        assert!(output.played.lock().unwrap().is_empty());
    }

    #[test]
    fn other_errors_skip_one_sentence() {
        let (mut queue, output) = fake_queue(|text| {
            if text.contains("بد") {
                Err(anyhow!("sidecar 500: boom"))
            } else {
                Ok(text.as_bytes().to_vec())
            }
        });
        queue.push("خوب است. بد است. باز هم بد. خوب شد.");
        queue.finish();
        wait_idle(&queue);
        assert_eq!(*output.played.lock().unwrap(), vec!["خوب است.", "خوب شد."]);
        let errors = queue.errors();
        assert_eq!(errors.try_recv().unwrap(), "sidecar 500: boom");
        assert!(
            errors.try_recv().is_err(),
            "the same error is reported once"
        );
    }

    #[test]
    fn unspeakable_text_is_skipped() {
        let (mut queue, output) = fake_queue(|text| Ok(text.as_bytes().to_vec()));
        queue.push("| a | b |\n|---|---|\n👍\n");
        queue.finish();
        assert!(!queue.is_active());
        thread::sleep(Duration::from_millis(50));
        assert!(output.played.lock().unwrap().is_empty());
    }
}
