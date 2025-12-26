import pandas as pd
import os
import logging

class TrendAgent:
    def __init__(self, storage_path='output/trend_report.csv'):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        # DataFrame columns: Date, Topic, Count
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(self.storage_path):
            try:
                return pd.read_csv(self.storage_path)
            except Exception:
                return pd.DataFrame(columns=['Date', 'Topic', 'Count'])
        return pd.DataFrame(columns=['Date', 'Topic', 'Count'])

    def update(self, date: str, topics: list[dict]):
        """
        Add a day's topics to the trend storage.
        """
        new_rows = []
        for t in topics:
            new_rows.append({
                'Date': date,
                'Topic': t['topic_label'],
                'Count': t['count']
            })
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            self.save()
            self.logger.info(f"Updated trends for {date} with {len(new_rows)} topics.")

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.data.to_csv(self.storage_path, index=False)

    def get_trend_matrix(self, start_date=None, end_date=None):
        """
        Returns a pivot table of Topic vs Date.
        """
        df = self.data.copy()
        if start_date:
            df = df[df['Date'] >= start_date]
        if end_date:
            df = df[df['Date'] <= end_date]
            
        # Ensure Count is numeric
        df['Count'] = pd.to_numeric(df['Count'], errors='coerce').fillna(0)
        
        pivot = df.pivot_table(index='Topic', columns='Date', values='Count', aggfunc='sum')
        pivot = pivot.fillna(0).infer_objects(copy=False)
        
        # Filter: Remove topics with 0 count in the selected window
        pivot = pivot[pivot.sum(axis=1) > 0]
        
        return pivot

    def clean_storage(self):
        self.data = pd.DataFrame(columns=['Date', 'Topic', 'Count'])
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
