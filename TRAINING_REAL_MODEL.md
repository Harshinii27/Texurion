# Training a real Texurion model

Texurion's image classifier uses real LBP + (optional) Gabor/GLCM features with a calibrated SVM/RF trained on your dataset.

## Dataset
- Dataset: `textures.zip` (Brodatz-like textures) has 3 classes: `banded`, `dotted`, `woven` (120 each, 360 total).
- Structure: `textures_upload1/<class>/<image>.jpg` (handled automatically).

## Train
```powershell
cd "C:\Users\harsh\Desktop\dip\labfile\Texture_Analysis_LBP(DIP-Project)"
.\venv314\Scripts\python.exe training\train_real.py
```
Outputs:
- `models/texurion_model.pkl` — full bundle (scaler, classifier, feature_config, classes, metrics)
- `models/texurion_model_meta.json` — reproducible metadata (splits, test metrics, version, date)

## Test reload
```powershell
.\venv314\Scripts\python.exe -c "
import joblib
from classifier import predict
from core.io import decode_image
from core.pipeline import FeatureConfig, extract_features
b=joblib.load('models/texurion_model.pkl')
img=decode_image(open('images/brick.jpeg','rb').read())
v=extract_features(img, FeatureConfig.from_dict(b['feature_config']))['vector']
print(predict(b,v))
"
```

## Notes
- Training uses stratified train/val/test splits (no leakage). 
- Features: multiscale LBP + Gabor (configurable). Uses `skimage` for consistency; matches inference.
- The model is loaded in UI via existing classifier/load path. If model is missing, UI shows honest fallback (rule-based descriptor).
- Reported test metrics: accuracy/precision/recall/F1 (weighted), confusion matrix, per-class via classification report in training output.
