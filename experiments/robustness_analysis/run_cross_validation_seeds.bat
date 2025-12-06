@echo off
REM ============================================================
REM  run_cross_validation_seeds.bat
REM  Runs 3 models × 3 modalities with repeated cross-validation
REM ============================================================

setlocal enabledelayedexpansion

REM ---- PATH SETTINGS ----
set ROOT=C:\Users\User\OneDrive\Documents\Thesis_Analysis_3_to_6\Robustness_analysis
set SCRIPT=%ROOT%\cross_validation_trainer.py
set DATA_ROOT=%ROOT%\data
set OUT_DIR=%ROOT%\cv_results
set PYTHON="C:\Program Files\Python312\python.exe"

REM ---- TRAINING CONFIG ----
set SPLITS=5
set REPEATS=5
set BASE=42
set BATCH=32
set LR=1e-3
set EPOCHS=5
set WORKERS=0

echo ============================================================
echo Starting Cross-Validation Experiments
echo Folder: %ROOT%
echo Base seed = %BASE%  |  Repeats = %REPEATS%  |  Folds = %SPLITS%
echo ============================================================
echo.

REM ---- Change to the correct working directory ----
cd /d "%ROOT%"

REM ======================
REM Run all combinations
REM ======================
for %%M in (resnet densenet efficientnet) do (
  for %%D in (ADC DWI T2) do (
    echo ----------------------------------------------------
    echo Running %%M on %%D ...
    echo ----------------------------------------------------

    %PYTHON% -u "%SCRIPT%" --model %%M --dataset %%D ^
      --data_root "%DATA_ROOT%" --out_dir "%OUT_DIR%" ^
      --n_splits %SPLITS% --n_repeats %REPEATS% ^
      --base_seed %BASE% --batch_size %BATCH% ^
      --lr %LR% --epochs %EPOCHS% --workers %WORKERS%

    echo.
  )
)

echo ============================================================
echo All cross-validation experiments completed!
pause
