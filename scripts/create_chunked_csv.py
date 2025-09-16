"""
Chunked CSV Creator
Creates chunked CSV files for large datasets to ensure reliable delivery.
"""

import pandas as pd
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkedCSVCreator:
    """Creates chunked CSV files for large datasets."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        self.data_dir = self.repo_root / "data"
        
        # Chunk size for CSV files
        self.chunk_size = 1000  # 1000 rows per chunk
    
    def create_chunked_trades_csv(self) -> dict:
        """Create chunked CSV from trades Parquet file."""
        logger.info("📊 Creating chunked trades CSV...")
        
        # Load trades from Parquet
        trades_path = self.reports_dir / "ledgers" / "trades.parquet"
        if not trades_path.exists():
            logger.error(f"Trades Parquet not found: {trades_path}")
            return {}
        
        df = pd.read_parquet(trades_path)
        total_rows = len(df)
        
        # Create chunked CSV files
        chunks_dir = self.reports_dir / "ledgers" / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        chunk_files = []
        num_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, total_rows)
            
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_file = chunks_dir / f"trades_chunk_{i+1:03d}_of_{num_chunks:03d}.csv"
            
            chunk_df.to_csv(chunk_file, index=False)
            chunk_files.append({
                'file': str(chunk_file.relative_to(self.repo_root)),
                'rows': len(chunk_df),
                'size_bytes': chunk_file.stat().st_size
            })
            
            logger.info(f"✅ Created chunk {i+1}/{num_chunks}: {chunk_file.name} ({len(chunk_df)} rows)")
        
        return {
            'source_file': str(trades_path.relative_to(self.repo_root)),
            'total_rows': total_rows,
            'chunk_size': self.chunk_size,
            'num_chunks': num_chunks,
            'chunk_files': chunk_files
        }


def main():
    """Main function to create chunked CSV files."""
    creator = ChunkedCSVCreator()
    
    # Create chunked trades CSV
    trades_chunks = creator.create_chunked_trades_csv()
    
    print("✅ Chunked CSV creation completed!")
    print(f"📊 Trades chunks: {trades_chunks.get('num_chunks', 0)} files")


if __name__ == "__main__":
    main()
