# tongflow-modal-levo

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text/lyrics-to-music generation with **LeVo 2 / SongGeneration** (`lglg666/SongGeneration-v2-large`), running on a GPU via [Modal](https://modal.com).

LeVo 2 is a commercial-grade, multilingual (zh, en, es, ja, …) song-generation model. It turns structured lyrics plus style tags into a full song with vocals and accompaniment.

## Capabilities

- **Music generation** (`gen-music`) — generate a song from lyrics + style tags.

### Input tips

- **Lyrics** use LeVo's structured format: sections separated by `;`, e.g.
  `[intro-short] ; [verse] First line. Second line. ; [chorus] Hook line. ; [outro-short]`.
  `[verse]`/`[chorus]`/`[bridge]` carry lyrics; `[intro-*]`/`[inst-*]`/`[outro-*]` are instrumental.
- **Style tags** go in the *tags* field as comma-separated keywords (gender, genre, emotion, instrument), e.g. `female, pop, warm, piano`. Avoid full sentences.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | — | Only needed if the LeVo weights become gated; public today. |

On first use the plugin downloads the weights (~13 GB checkpoint + runtime bundle) to your Modal `models` volume, deploys the app to your Modal account, and caches the build. Runs on an L40S GPU.
