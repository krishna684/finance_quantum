"""TikTok financial content collection pipeline.

Replaces the TikTok Research API collector with a Bright Data scraping
architecture adapted from the prediction-markets pipeline:

    Bright Data keyword search
      → Pass 1 regex filter (cashtags / ticker names / finance genre)
      → Pass 2 LLM YES/NO topic filter (Anthropic)
      → yt-dlp video download + OpenAI Whisper transcription
      → ffmpeg frame extraction → CLIP image embeddings
      → FinBERT text embeddings
      → Financial entity/stance extraction (Anthropic)
      → INFORMATION vs TRADING content classification (Anthropic)
      → Storage in tiktok_fin.db (plugs into ClassificationPipeline)
"""
