//! Pool-owned spawn thread. Linux PDEATHSIG follows the creating thread's lifetime.
use std::{
    io,
    process::{Child, Command},
    sync::mpsc,
    thread,
};
struct Request {
    command: Command,
    reply: mpsc::SyncSender<io::Result<Child>>,
}
pub struct ChildLauncher {
    requests: Option<mpsc::Sender<Request>>,
    owner: Option<thread::JoinHandle<()>>,
}
impl ChildLauncher {
    pub fn new() -> io::Result<Self> {
        let (requests, receiver) = mpsc::channel::<Request>();
        let owner = thread::Builder::new()
            .name("machine-child-launcher".into())
            .spawn(move || {
                // Abandoned pre-handshake children have not received authored Start/Invoke.
                // Keep/reap their handles rather than silently orphaning them. The thread
                // only spawns: handshake, model load and computation remain on the caller.
                let mut abandoned: Vec<Child> = vec![];
                while let Ok(mut request) = receiver.recv() {
                    abandoned.retain_mut(|child| !matches!(child.try_wait(), Ok(Some(_))));
                    let child = request.command.spawn();
                    if let Err(mpsc::SendError(Ok(child))) = request.reply.send(child) {
                        abandoned.push(child);
                    }
                }
                // Ending the pool-owned thread ends its pre-CUDA ownership boundary.
                // Normal pool.stop observes managed children exit before this drop.
            })?;
        Ok(Self {
            requests: Some(requests),
            owner: Some(owner),
        })
    }
    pub fn spawn(&self, command: Command) -> io::Result<Child> {
        let (reply, result) = mpsc::sync_channel(1);
        self.requests
            .as_ref()
            .ok_or_else(|| io::Error::other("child launcher closed"))?
            .send(Request { command, reply })
            .map_err(|_| io::Error::other("child launcher ended"))?;
        result
            .recv()
            .map_err(|_| io::Error::other("child launcher reply lost"))?
    }
}
impl Drop for ChildLauncher {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(owner) = self.owner.take() {
            let _ = owner.join();
        }
    }
}
