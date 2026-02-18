"""Full text indexing using Whoosh."""
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import logging
from whoosh.index import create_in, exists_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.query import And, Term
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """A search result."""
    page_num: int
    content: str
    score: float

class IndexManager:
    """Manage full text search index."""

    def __init__(self, index_dir: Optional[Path] = None):
        """Initialize index manager."""
        self.index_dir = index_dir or Path(tempfile.mkdtemp())
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self._init_index()

    def _init_index(self) -> None:
        """Initialize Whoosh index."""
        if exists_in(str(self.index_dir)):
            self.index = open_dir(str(self.index_dir))
        else:
            self.index = create_in(
                str(self.index_dir),
                schema=Schema(
                    doc_id=ID(stored=True),
                    page_num=ID(stored=True),
                    content=TEXT(stored=True)
                )
            )

    def index_document(self, doc_id: str, pages: List[tuple]) -> None:
        """Index a document.

        Args:
            doc_id: Document identifier
            pages: List of (page_num, content) tuples
        """
        writer = self.index.writer()

        # Clear existing document
        writer.delete_by_term("doc_id", doc_id)

        # Add pages
        for page_num, content in pages:
            writer.add_document(
                doc_id=doc_id,
                page_num=str(page_num),
                content=content
            )

        writer.commit()
        logger.info(f"Indexed document {doc_id} with {len(pages)} pages")

    def search(self, query: str, doc_id: Optional[str] = None, limit: int = 10) -> List[SearchResult]:
        """Search the index.

        Args:
            query: Search query string
            doc_id: Optional document ID to restrict search
            limit: Maximum number of results

        Returns:
            List of SearchResult
        """
        parser = QueryParser("content", self.index.schema)
        q = parser.parse(query)

        if doc_id:
            # Use And query to combine content search with doc_id filter
            q = And([q, Term("doc_id", doc_id)])

        with self.index.searcher() as searcher:
            results = searcher.search(q, limit=limit)

            search_results = []
            for r in results:
                search_results.append(SearchResult(
                    page_num=int(r["page_num"]),
                    content=r.get("content", "")[:200],  # Preview
                    score=r.score
                ))

            return search_results

    def clear_document(self, doc_id: str) -> None:
        """Remove a document from index."""
        writer = self.index.writer()
        writer.delete_by_term("doc_id", doc_id)
        writer.commit()

    def close(self) -> None:
        """Close the index."""
        if self.index:
            self.index.close()
