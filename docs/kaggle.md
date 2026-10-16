# Running on Kaggle

The repository is private, so the notebooks clone it using a Kaggle secret.

## One-time setup

1. On GitHub, open Settings -> Developer settings -> Personal access tokens -> Fine-grained tokens -> Generate new token.
   Set **Resource owner** to `brainardphilemon`, choose **Only select repositories** -> `hallucination-probes`, and under
   Repository permissions set **Contents: Read-only** (Metadata read is added automatically). Pick an expiry.
2. In a Kaggle notebook, open Add-ons -> Secrets -> Add secret. Name it `GITHUB_TOKEN`, paste the token, and tick the
   checkbox that attaches it to the notebook. You need to attach it again for every new notebook, including copies.
3. Settings: Accelerator **GPU T4 x2**, Internet **on**.

## Extract (`notebooks/kaggle_extract.ipynb`)

1. Set `DOMAINS` and `ARTIFACT_NAME` in the config cell.
2. Click Save Version → Save & Run All.
3. The output folder `/kaggle/working/<ARTIFACT_NAME>` contains `generation/` and `bleurt/`.

The full 7-dataset extraction takes several hours. Runs resume, so a timed-out run can be continued.

## Train (`notebooks/kaggle_train.ipynb`)

1. Attach the extract notebook's output, or the existing `sala_qwen25_g14_6src_triviaqa_v1` artifacts, as input. Both use the same layout.
2. Set `ARTIFACT_NAME`, `SOURCES` and `TARGETS`.
3. Add one entry per experiment to `RUNS`. Each entry is a list of `hprobes train` arguments, as described in the README.
4. Click Save Version → Save & Run All.

The last cell prints a table comparing all runs. The per-run files are in `/kaggle/working/runs/<name>/`.

For reproducible runs, set `REF` in the first cell to a commit hash instead of `main`.

## Troubleshooting

* `git clone failed; GitHub API answered 401`: the token is invalid, expired or was revoked. Create a new one.
* `... answered 404`: the token cannot see the private repository. Most often it was created for a different repository
  or owner, or without Contents: Read-only.
* `Secret not found` / `No secrets`: the secret is not named exactly `GITHUB_TOKEN` or is not attached to this notebook.
* The setup cell passes the token to git only through an environment header, so it never appears in the command, URL,
  traceback or notebook log. If you ever see a token in a log, revoke it at once.
