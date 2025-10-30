@echo off
echo ============================================================
echo   Training Cancer Detection Model
echo ============================================================
echo.
echo This will train the model with your 10,000 labeled images.
echo Training may take 1-3 hours depending on your hardware.
echo.
echo Configuration:
echo   - Images: 10,000 chest X-rays
echo   - Labels: Binary (Cancer / No Cancer)
echo   - Model: ResNet50 (pretrained)
echo   - Epochs: 100 (with early stopping)
echo   - Batch size: 16
echo.
pause

echo.
echo Starting training...
echo.

python main.py train --config configs/config.yaml

echo.
echo ============================================================
echo   Training Complete!
echo ============================================================
echo.
echo The trained model is saved in:
echo   experiments/[experiment_name]/checkpoints/best_model.pth
echo.
echo You can now use this model for real predictions!
echo.
pause
