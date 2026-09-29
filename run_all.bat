@echo off
setlocal
REM Set PY to the Python interpreter of your environment if "python" is not on PATH.
set PY=python
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist results mkdir results

REM ---- Step 1: build frame/KTS cache + random/human baselines + decode timing (once per dataset)
%PY% evaluate_cv.py --dataset summe --baselines_only --time_decode || goto :error
%PY% evaluate_cv.py --dataset tvsum --baselines_only --time_decode || goto :error

REM ---- Step 2: train (last-epoch checkpoints, no test-fold selection) + evaluate
REM Seed is the outer loop, so after the first pass every configuration already has one result.
for %%S in (42 43 44) do (
  for %%D in (summe tvsum) do (
    for %%B in (mobilevitv2_050 mobilevit_xxs) do (
      for %%C in (proposed pure surgery_only full_temporal) do (
        %PY% train_cv.py --dataset %%D --backbone %%B --config %%C --seed %%S || goto :error
        %PY% evaluate_cv.py --dataset %%D --backbone %%B --config %%C --seed %%S || goto :error
      )
    )
  )
  %PY% summarize_results.py > results\summary_after_seed_%%S.md
)

REM ---- Step 3: final tables
%PY% summarize_results.py > results\summary.md
echo Done. See results\summary.md
goto :eof

:error
echo [!] Stopped because of an error. Re-running this file resumes from where it stopped.
exit /b 1
