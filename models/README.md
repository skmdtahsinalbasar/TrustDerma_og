# Models Folder

Place your trained model file here:

```
best_baseline.pth
```

## Expected Model Details

- **Architecture:** EfficientNetV2-S (`tf_efficientnetv2_s`)
- **Input Size:** 384 x 384
- **Number of Classes:** 7
- **Class Order:** `["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]`
- **State Dict Key:** Either the raw state dict directly, or stored under `"model_state_dict"` key in a checkpoint

The app will look for the model at `models/best_baseline.pth` by default.
You can override this with the `TRUSTDERMA_MODEL` environment variable.
