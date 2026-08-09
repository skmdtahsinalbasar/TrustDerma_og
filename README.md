# TrustDerma Web App

## Run
1. Put your existing `best_baseline.pth` in `models/`.
2. Install:
`pip install -r requirements.txt`
3. Start:
`streamlit run app.py`

The app will open automatically at: **http://localhost:8501**

---

##  Project Structure
```
TrustDerma-2/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Ignore patterns
├── vercel.json                  # Vercel deploy config (⚠️ not recommended)
├── Procfile                     # Heroku/Railway web process
├── setup.sh                     # Streamlit runtime config bootstrapper
├── runtime.txt                  # Python 3.11.4 pin (Heroku)
├── railway.json                 # Railway.app auto-config
├── render.yaml                  # Render.com 1-click Blueprint
├── packages.txt                 # Streamlit Cloud system libs
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── models/
    ├── .gitkeep                # Keep folder in repo
    ├── README.md               # Model instructions
    └── best_baseline.pth       # YOUR MODEL (not in repo - add locally)
```

---

##  Configuration

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `TRUSTDERMA_MODEL` | `models/best_baseline.pth` | Path to your trained model checkpoint |

### Model Format
The checkpoint can be saved as either:
- **Raw state dict:** `torch.save(model.state_dict(), "best_baseline.pth")`
- **Full checkpoint:** `torch.save({"model_state_dict": model.state_dict(), ...}, "best_baseline.pth")`

---

##  Troubleshooting

### PermissionError: `.streamlit` directory
Create the directory manually:
```bash
mkdir -p ~/.streamlit
mkdir -p .streamlit
```
Or disable usage stats in `.streamlit/config.toml`:
```toml
[browser]
gatherUsageStats = false
```

### ValueError: operands could not be broadcast together (Grad-CAM shapes)
This is fixed in the current version of `app.py`. If you see it, ensure `cv2.resize()` is being called on the heatmap before overlaying.

### Model not loading
- Verify the file exists at `models/best_baseline.pth`
- Use absolute path or override with `TRUSTDERMA_MODEL=/full/path/best_baseline.pth`
- Ensure the architecture is `tf_efficientnetv2_s` with 7 output classes

---

##  Deployment Guide

###  Quick Platform Comparison

| Platform | Cost | Works with Streamlit? | Difficulty | Best For |
|---|---|---|---|---|
| **Streamlit Community Cloud** | **FREE** | ✅ **Native & Official** | ⭐ Very Easy | **✅ RECOMMENDED** |
| **Render.com** | Free tier (sleeps) + $7/mo always-on | ✅ Perfect | ⭐ Easy | Production + Free tier |
| **Railway.app** | $5+/month (trial credits) | ✅ Perfect | ⭐ Easy | Production, fast deploys |
| **Heroku** | $7+/month | ✅ Works | ⭐ Easy | Legacy production |
| **Vercel** | Free tier | ⚠️ **Not Recommended** (WebSocket issues, no persistence) | ⭐⭐⭐ Hard | Next.js apps ONLY |

---

###  1. Streamlit Community Cloud (FREE, EASIEST, RECOMMENDED)

Built specifically for Streamlit apps. Zero config, automatic GitHub deploys.

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub → **New app**
3. Configure:
   - **Repository:** `skmdtahsinalbasar/TrustDerma-2`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings** → **Python version:** `3.11`
5. **For your model file `best_baseline.pth`:**
   - Option A: Add via **Secrets → TOML format:** paste download script that fetches the model at runtime
   - Option B: Use `TRUSTDERMA_MODEL` env var pointing to a public direct download link (Hugging Face Hub, Dropbox, GDrive direct)
6. Click **Deploy**

Your live URL: `https://skmdtahsinalbasar-trustderma-2.streamlit.app`

---

###  2. Render.com (Great Free Tier)

Free tier available (sleeps after inactivity), paid starts at $7/mo always-on.

`render.yaml` file is **already included** for 1-click deploy!

#### Option A: 1-Click using render.yaml
1. Go to **https://dashboard.render.com/blueprints**
2. **New from YAML** → paste your repo URL
3. Review & click **Create Resources**

#### Option B: Manual Web Service
1. Go to **https://render.com** → **New + → Web Service**
2. Connect GitHub → pick `TrustDerma-2` repo
3. Configure:
   - **Name:** `trustderma`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:**
     ```bash
     sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
     ```
4. Under **Environment Variables** (optional):
   - `TRUSTDERMA_MODEL` = public download URL of your `best_baseline.pth`
   - Or copy/paste the file into the project via Render shell after deploy
5. Click **Create Web Service**

---

###  3. Railway.app (Fast, $5/mo, has trial credits)

`railway.json` config file is **already included**!

1. Go to **https://railway.app** → Sign in with GitHub
2. **New Project** → **Deploy from GitHub repo**
3. Select `TrustDerma-2` repo → **Deploy Now**
4. Railway auto-reads `railway.json` and configures everything
5. Add env var (if needed):
   - Go to **Variables** tab → Add `TRUSTDERMA_MODEL`
6. Generate a public domain: **Settings → Networking → Generate Domain**

---

###  4. Heroku ($7+/month)

Files `Procfile` + `setup.sh` + `runtime.txt` already included!

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login & create app
heroku login
heroku create trustderma-app

# Deploy
git push heroku main

# Open your app
heroku open
```

To add the model file, set env var:
```bash
heroku config:set TRUSTDERMA_MODEL="https://your-storage.com/best_baseline.pth"
```

---

###  5. Vercel (⚠️ NOT Recommended for Streamlit)

> **Important Caveat:** Vercel is built for **Next.js / static / serverless (request-response)** workloads.
> Streamlit uses **persistent WebSocket connections + in-memory state**, which Vercel Serverless Functions don't handle well.
> You will see random connection drops, state loss, and upload failures.
> **USE Streamlit Community Cloud, Render, or Railway instead for reliable operation.**

#### If you still want to try Vercel:
`vercel.json` config file is already included.

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```
2. Deploy from project root:
   ```bash
   cd TrustDerma_Web
   vercel
   ```
3. Follow prompts → connect project
4. In Vercel dashboard → **Project Settings → Functions**:
   - Memory: **1024 MB** (minimum for PyTorch)
   - Max Duration: **60s**
5. Add env var `TRUSTDERMA_MODEL` pointing to a hosted model URL
6. **Expect WebSocket errors and intermittent issues** → strongly consider one of the recommended platforms above.

---

###  Hosting your model file (best practices)

Since `best_baseline.pth` is too large for raw GitHub uploads, host it publicly on:
- **Hugging Face Hub** (free, best for ML models) → create a model repo, upload `.pth`, get direct URL
- **Google Drive** → get a **direct download link** (via `https://drive.google.com/uc?export=download&id=FILE_ID`)
- **Dropbox** → create shared link, replace `www.dropbox.com` with `dl.dropboxusercontent.com`
- **GitHub Releases** → upload as release asset (handles large files)

Then set env var on your deployment platform:
```
TRUSTDERMA_MODEL=https://huggingface.co/your-username/your-repo/resolve/main/best_baseline.pth
```

---

##  Medical Disclaimer

**This is a research/decision-support prototype. It does NOT establish a definitive medical diagnosis and must NOT replace professional medical assessment, dermoscopy, biopsy, or clinical judgment.**

MC Dropout is not used as the primary Trust Score because the original experiment showed essentially zero stochastic variation.
