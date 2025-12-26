# Senior AI Engineer Assignment – App Review Trend Analysis


## Problem Statement
App store reviews are a goldmine of user feedback, but manually tracking trends across thousands of daily reviews is impossible. Product teams need a way to see "what's trending" (e.g., "Login failing after update") in real-time. This system uses AI agents to process reviews, extract topics, and track their volume over a rolling 30-day window.

## Architecture Overview
The system is built as a pipeline of specialized agents:

1.  **Ingestion**: Loads reviews (Simulated for this demo to guarantee trend visibility).
2.  **Topic Extraction Agent**: Uses Sentence Embeddings (all-MiniLM-L6-v2) and Clustering to group reviews into semantic topics without needing pre-defined keywords.
3.  **Deduplication Agent**: Maps new extracted topics to a canonical "Topic Memory" to ensure "App crashing" and "Crahsed app" are counted as the same trend.
4.  **Evolution Agent**: Identifies truly new topics that haven't been seen before and registers them.
5.  **Trend Agent**: Aggregates counts and produces the final rolling window report.
The final output is a tabular trend report where rows represent canonical topics and columns represent dates from T-30 to T. Each cell contains the frequency of that topic on the given day. The report is exported as a CSV for easy consumption by product teams.


## Design Decisions
-   **Recall over Precision**: We prefer to surface a potential issue even if the label is slightly noisy, rather than burying a critical bug.
-   **Explainability**: Every specific review contributes to a topic count. We can trace a trend spike back to individual user quotes.
-   **No Black Box**: We use explicit clustering and similarity thresholds, avoiding opaque end-to-end LLM calls for the core counting loop to ensure speed and low cost.

## Assumption & Limitations
-   **Sarcasm**: Sentiment is not deeply analyzed; "Great job breaking the app" might be clustered with "App broken".
-   **Context**: Short reviews like "Bad" are filtered or ignored as they don't provide actionable topics.

## Setup
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the demo directly: `python demo.py`
