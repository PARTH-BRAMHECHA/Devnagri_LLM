# Running `devanagri-llm-hindi-only.ipynb` on Kaggle across multiple sessions

This notebook now handles two problems that previously caused silent failures:

1. **Session time limit** — Kaggle kills a kernel at a hard 12h wall-clock cap,
   with no exception raised. The notebook now tracks its own elapsed time and
   stops Stage 2d / refuses to start Stage 3 *before* that happens.
2. **Cross-session persistence** — `/kaggle/working` does not survive between
   separate Kaggle sessions unless you explicitly commit the notebook. The
   notebook now restores its checkpoint state from a previous committed run,
   if you attach that run's Output as an input.

Because a single run of Stage 2d can take longer than 12h, **you will likely
need multiple runs** to get through the full pipeline. This guide is the loop
you repeat until it's done.

---

## The one rule that matters

**Never run this notebook by pressing "Run" cell-by-cell in an interactive
Edit session and walking away.** Interactive sessions do not save anything —
if the browser tab closes, disconnects, or the 12h cap hits, everything in
`/kaggle/working` (including checkpoints) is gone.

Always use **Save Version → Save & Run All (Commit)**. Only a commit's final
`/kaggle/working` state is preserved, as that version's **Output**.

---

## First run (starting fresh)

1. Open the notebook on Kaggle.
2. Confirm the GPU accelerator (T4) and internet access are turned on
   (Settings panel, right sidebar).
3. Click **Save Version** → choose **Save & Run All (Commit)** → **Save**.
4. Kaggle runs the whole notebook top to bottom in the background. You can
   close the tab; it keeps running.
5. Wait for the commit to finish (Kaggle emails you, or check the
   notebook's **Versions** tab). Open the version's logs and look at the
   final cell's output ("FINAL SANITY SUMMARY"):
   - If it prints **`✓ PIPELINE COMPLETE`** — you're done, skip to
     [Reading your results](#reading-your-results).
   - If it prints **`⚠ PIPELINE NOT YET COMPLETE`** — continue to the next
     section. This is expected and not an error; Stage 2d likely needed more
     than one 12h session.

---

## Every subsequent run (resuming)

This is the step that makes checkpoints actually carry forward. Do this
**before** each new commit after the first one:

1. Open the notebook in **Edit** mode.
2. In the right sidebar, click **Add Input**.
3. Go to the **Notebook Output Files** tab (not "Datasets" or "Models").
4. Search for this same notebook by name.
5. Select it — Kaggle attaches its **latest committed version's Output** as
   a read-only input, mounted at `/kaggle/input/<notebook-slug>/`.
6. Click **Save Version → Save & Run All (Commit)** again.

That's it. The new run's first real cell — **"0b. RESTORE CHECKPOINT STATE
FROM A PREVIOUS SESSION"** — automatically finds the attached
`persistent/` folder and copies it back into `/kaggle/working/persistent`
before Stage 2d starts, so training resumes from the last saved step instead
of restarting at 0.

Look at that cell's output in the run log to confirm it worked:

```
↺ Found checkpoint state from a previous session: /kaggle/input/<slug>/persistent
✓ Restored persistent state into /kaggle/working/persistent
  Finished Stage 2d checkpoints found: []
  Finished Stage 3 results found:      []
```

(An empty list just means Stage 2d hasn't reached `final` yet — intermediate
`step_NNNN` checkpoints still restore and resume correctly; only the guard
checks look for `final` specifically.)

**Repeat this section** — commit, check the summary, add the latest Output as
input, commit again — until the final cell prints `✓ PIPELINE COMPLETE`.

> **Note:** you only need to re-attach the input once you have a *newer*
> version to point at. If Kaggle's "Add Input" UI lets you pick "always use
> latest version" for this notebook's own output, prefer that — it saves you
> from manually re-adding it every single time.

---

## How to tell what a run actually accomplished

Every run's final cell prints a status table. Read it top to bottom:

```
✓ Repository
✓ Hindi train
✓ Hindi test
✗ Hindi Stage 2d final        <- still training; will resume next run
✗ Hindi Stage 3 results       <- can't start until the line above is ✓
✗ Hindi Stage 4 directory
```

If Stage 2d's own loop had to stop early because the session was running out
of time, you'll also see this earlier in the log (from the Stage 2d cell):

```
⏱ Stopping Stage 2d for hindi after round 3: only 0.41h left in this
Kaggle session (12.0h cap - 1.5h reserved for Stage 3/4), but another
round needs ~3.25h. The last checkpoint is already saved under
/kaggle/working/persistent/models/devaware_finetuned/hindi, so it's
safe to stop here.
```

This is expected, not a failure — it's the fix working as intended. Just
follow the resume steps above for the next run.

---

## Reading your results

Once `✓ PIPELINE COMPLETE` prints, your outputs are in that version's
Output tab:

- `persistent/models/devaware_finetuned/hindi/final/` — the fine-tuned
  checkpoint
- `persistent/results/hindi/compression_results.json` — Stage 3's
  three-condition comparison
- `persistent/logs/` — full run logs for every stage

You can download these directly from the Output tab, or attach that version
as an input to a separate analysis notebook.

---

## FAQ

**Do I need to change any code between runs?**
No. `LANGUAGES`, `HOURS_PER_ROUND`, `MAX_ROUNDS`, and the session-budget
constants are all designed to just keep working across repeated commits.

**What if I forget to attach the previous Output as an input?**
The restore cell prints `"No previous session's Output is attached..."` and
the run starts from scratch — Stage 2d will retrain from step 0. Nothing
breaks, but you lose the previous run's progress. Always double-check the
restore cell's log output before assuming a run resumed correctly.

**Can I run this interactively to debug a single cell?**
Yes, for quick checks (e.g. verifying the dataset path). Just don't rely on
that session for real training progress — commit whenever you want a run's
work to actually survive.

**How many runs will the full pipeline take?**
Based on the reference run: Stage 2d for hindi alone took ~12h across 4
rounds (essentially a full session). Budget at least 2 sessions per
language for Stage 2d + Stage 3/4, more if Kaggle's weekly GPU quota
(~30h) runs out first — in which case you'll need to wait for the weekly
reset between commits.

**Does this work for `marathi` and `sanskrit` too?**
Yes — the restore/persistence logic is language-agnostic. If you extend
`LANGUAGES` beyond `["hindi"]`, the same restore-cell and persistent-storage
fix carries every language's checkpoints and results forward the same way.
