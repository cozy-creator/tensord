# Rental lifetime

Owner ruling (Paul, 2026-10-10, verbatim): "The way it should work is that rentals kill themselves
after being idle for 15 minutes (that is, they are not processing any jobs). If the H100 killed
itself mid-run, it was clearly not idle; it killed itself for some other stupid reason. I never
created some random new rule." And: "1. if you have a job to work on, you are not idle. You are
trying to work on that job. 2. if you're working on a job, you're not idle. 3. if you do not have a
job to work on, you're idle. … A machine doing package preparation is idle unless it has a queued
job it's going to work on."

- Every rental, used or not, releases itself after 15 minutes with no job queued for it or running
  on it. There is no "used rentals never expire" rule.
- A job is any run but a warm-up: queued (preparing included), starting or running. Warm-ups,
  uploads, updates and paused jobs are idle time.
- The clock starts when the last job ends (or pauses), or at the rental's first boot, or at an
  explicit `cozy rental keepalive`, which restarts it once.
- The clock is durable: the journal's last job end and the ledger's idle start (`idle.json`). A
  restart or update neither shortens nor extends it; a job a restart interrupts ends at the restart.
- Status reports the real deadline: now + 15 minutes while a job is queued or running, else the
  clock's end. 0 only for a machine that never releases itself.
