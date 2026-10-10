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
- A job is any run asked of the machine (a call, a job, a warm-up, a model upload, a conversion or
  quantization): queued (preparing included), starting or running. Each ends completed or failed.
  Updates and paused jobs are idle time.
- The clock starts when the last job ends (or pauses), or at the rental's first boot, or at an
  explicit `cozy rental keepalive`, which restarts it once.
- The clock is durable: the journal's last job end and the ledger's idle start (`idle.json`). A
  restart or update neither shortens nor extends it; a job a restart interrupts ends at the restart.
- Status reports a deadline only once the rental is idle: the clock's end. While a job is queued
  or running there is none (0), as for a machine that never releases itself.
