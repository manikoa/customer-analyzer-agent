"""
Data Ingestion Module
Handles importing customer feedback from various sources:
- CSV files (reviews, surveys)
- JSON files (support tickets, API data)
- Direct data structures
"""

import pandas as pd
import json
from datetime import datetime
import os


class DataIngestion:
    """Handles data ingestion from multiple customer feedback sources."""
    
    @staticmethod
    def load_from_csv(file_path, text_column='text', date_column='date', source_column='source'):
        """
        Load feedback data from CSV file.
        
        Args:
            file_path: path to CSV file
            text_column: name of column containing feedback text
            date_column: name of column containing dates (optional)
            source_column: name of column containing source info (optional)
        
        Returns:
            pandas DataFrame with standardized columns
        """
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df = df.rename(columns={text_column: 'text'})
        
        if date_column in df.columns:
            df = df.rename(columns={date_column: 'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = datetime.now()
        
        if source_column in df.columns:
            df = df.rename(columns={source_column: 'source'})
        else:
            df['source'] = os.path.basename(file_path)
        
        return df[['text', 'date', 'source']]
    
    @staticmethod
    def load_from_json(file_path, text_field='text', date_field='date', source_field='source'):
        """
        Load feedback data from JSON file.
        
        Args:
            file_path: path to JSON file
            text_field: name of field containing feedback text
            date_field: name of field containing dates (optional)
            source_field: name of field containing source info (optional)
        
        Returns:
            pandas DataFrame with standardized columns
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list of objects and single object
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)
        
        # Standardize column names
        if text_field in df.columns:
            df = df.rename(columns={text_field: 'text'})
        
        if date_field in df.columns:
            df = df.rename(columns={date_field: 'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = datetime.now()
        
        if source_field in df.columns:
            df = df.rename(columns={source_field: 'source'})
        else:
            df['source'] = os.path.basename(file_path)
        
        return df[['text', 'date', 'source']]
    
    @staticmethod
    def load_reviews(file_path):
        """
        Load customer reviews from file.
        Supports CSV and JSON formats.
        
        Args:
            file_path: path to reviews file
        
        Returns:
            pandas DataFrame with review data
        """
        if file_path.endswith('.csv'):
            df = DataIngestion.load_from_csv(
                file_path,
                text_column='review',
                date_column='date',
                source_column='platform'
            )
        elif file_path.endswith('.json'):
            df = DataIngestion.load_from_json(
                file_path,
                text_field='review_text',
                date_field='review_date',
                source_field='platform'
            )
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        df['source'] = df['source'].fillna('review')
        return df
    
    @staticmethod
    def load_survey_responses(file_path):
        """
        Load survey responses from file.
        
        Args:
            file_path: path to survey responses file
        
        Returns:
            pandas DataFrame with survey data
        """
        if file_path.endswith('.csv'):
            df = DataIngestion.load_from_csv(
                file_path,
                text_column='response',
                date_column='submission_date',
                source_column='survey_name'
            )
        elif file_path.endswith('.json'):
            df = DataIngestion.load_from_json(
                file_path,
                text_field='response_text',
                date_field='submitted_at',
                source_field='survey_id'
            )
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        df['source'] = 'survey_' + df['source'].astype(str)
        return df
    
    @staticmethod
    def load_support_tickets(file_path):
        """
        Load support ticket data from file.
        
        Args:
            file_path: path to support tickets file
        
        Returns:
            pandas DataFrame with ticket data
        """
        if file_path.endswith('.csv'):
            df = DataIngestion.load_from_csv(
                file_path,
                text_column='ticket_description',
                date_column='created_date',
                source_column='ticket_type'
            )
        elif file_path.endswith('.json'):
            df = DataIngestion.load_from_json(
                file_path,
                text_field='description',
                date_field='created_at',
                source_field='category'
            )
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        df['source'] = 'support_ticket'
        return df
    
    @staticmethod
    def combine_sources(*dataframes):
        """
        Combine multiple feedback sources into a single DataFrame.
        
        Args:
            *dataframes: variable number of DataFrames to combine
        
        Returns:
            combined pandas DataFrame
        """
        if not dataframes:
            return pd.DataFrame(columns=['text', 'date', 'source'])
        
        combined = pd.concat(dataframes, ignore_index=True)
        combined = combined.drop_duplicates(subset=['text'], keep='first')
        combined = combined.sort_values('date', ascending=False)
        combined = combined.reset_index(drop=True)
        
        return combined


# Example usage
if __name__ == "__main__":
    print("Data Ingestion Module - Example Usage\n")
    
    # Example 1: Load from CSV
    sample_csv_data = pd.DataFrame({
        'review': ['Great product!', 'Not satisfied'],
        'date': ['2024-01-01', '2024-01-02'],
        'platform': ['Amazon', 'Google']
    })
    sample_csv_data.to_csv('sample_reviews.csv', index=False)
    
    reviews = DataIngestion.load_reviews('sample_reviews.csv')
    print("Loaded reviews:")
    print(reviews)
    print()
    
    # Clean up
    os.remove('sample_reviews.csv')
    
    print("✓ Data ingestion module ready to use!")

