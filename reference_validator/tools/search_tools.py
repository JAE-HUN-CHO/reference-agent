"""
검색 도구 모듈 - Tavily 및 Crawl 지원

레퍼런스 검증을 위한 웹 검색 기능을 제공합니다.
"""

import os
import re
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SearchResult


# ============================================================================
# 검색 도구 베이스 클래스
# ============================================================================

class BaseSearchTool(ABC):
    """검색 도구 베이스 클래스"""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """검색 수행"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """도구 사용 가능 여부 확인"""
        pass


# ============================================================================
# Tavily 검색 도구
# ============================================================================

class TavilySearchTool(BaseSearchTool):
    """Tavily API를 사용한 검색 도구"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = None
    
    def is_available(self) -> bool:
        """Tavily API 사용 가능 여부 확인"""
        if not self.api_key:
            return False
        try:
            from tavily import TavilyClient
            return True
        except ImportError:
            return False
    
    def _get_client(self):
        """Tavily 클라이언트 반환"""
        if self._client is None:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)
        return self._client
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Tavily를 사용하여 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            SearchResult 리스트
        """
        if not self.is_available():
            print("Warning: Tavily API가 설정되지 않았습니다.")
            return []
        
        try:
            client = self._get_client()
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",  # 더 깊은 검색
                include_raw_content=True,  # 원본 콘텐츠 포함
            )
            
            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    source="tavily",
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    content=item.get("raw_content", None),
                    score=item.get("score", 0.0),
                ))
            
            return results
            
        except Exception as e:
            print(f"Tavily 검색 오류: {e}")
            return []
    
    def search_academic(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        학술 논문 검색에 최적화된 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            SearchResult 리스트
        """
        # 학술 데이터베이스를 우선 검색하도록 쿼리 수정
        academic_query = f"{query} site:scholar.google.com OR site:semanticscholar.org OR site:arxiv.org OR site:doi.org"
        return self.search(academic_query, max_results)


# ============================================================================
# 크롤러 도구
# ============================================================================

class CrawlerTool(BaseSearchTool):
    """URL 크롤링 도구"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def is_available(self) -> bool:
        """크롤러 사용 가능 여부"""
        return True
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        쿼리가 URL인 경우 해당 URL을 크롤링합니다.
        쿼리가 URL이 아닌 경우 빈 리스트를 반환합니다.
        """
        if self._is_url(query):
            result = self.crawl_url(query)
            return [result] if result else []
        return []
    
    def _is_url(self, text: str) -> bool:
        """텍스트가 URL인지 확인"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(text))
    
    def crawl_url(self, url: str) -> Optional[SearchResult]:
        """
        URL에서 콘텐츠를 크롤링합니다.
        
        Args:
            url: 크롤링할 URL
            
        Returns:
            SearchResult 또는 None
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            content = response.text
            
            # HTML에서 제목 추출
            title = self._extract_title(content)
            
            # HTML에서 본문 텍스트 추출
            text_content = self._extract_text(content)
            
            # 스니펫 생성 (처음 500자)
            snippet = text_content[:500] if text_content else ""
            
            return SearchResult(
                source="crawl",
                title=title,
                url=url,
                snippet=snippet,
                content=text_content,
                score=1.0,  # 직접 크롤링이므로 관련도 최고
            )
            
        except Exception as e:
            print(f"URL 크롤링 오류 ({url}): {e}")
            return None
    
    def _extract_title(self, html: str) -> str:
        """HTML에서 제목 추출"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            return title_tag.get_text().strip() if title_tag else ""
        except ImportError:
            # BeautifulSoup 없으면 정규식으로 추출
            match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""
    
    def _extract_text(self, html: str) -> str:
        """HTML에서 본문 텍스트 추출"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 스크립트, 스타일 태그 제거
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            # 연속 공백 정리
            text = re.sub(r'\s+', ' ', text)
            return text
            
        except ImportError:
            # BeautifulSoup 없으면 간단한 태그 제거
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()


# ============================================================================
# 학술 데이터베이스 검색 도구
# ============================================================================

class SemanticScholarTool(BaseSearchTool):
    """Semantic Scholar API를 사용한 검색 도구"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
    
    def is_available(self) -> bool:
        """API 사용 가능 여부 (무료 티어도 가능)"""
        return True
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Semantic Scholar에서 논문을 검색합니다.
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            SearchResult 리스트
        """
        try:
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,url,venue,citationCount"
            }
            
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for paper in data.get("data", []):
                # 저자 문자열 생성
                authors = ", ".join([
                    a.get("name", "") for a in paper.get("authors", [])[:3]
                ])
                if len(paper.get("authors", [])) > 3:
                    authors += " et al."
                
                # 스니펫 생성
                snippet = paper.get("abstract", "")[:300] if paper.get("abstract") else ""
                if snippet:
                    snippet = f"[{paper.get('year', 'N/A')}] {authors}. {snippet}..."
                
                results.append(SearchResult(
                    source="semantic_scholar",
                    title=paper.get("title", ""),
                    url=paper.get("url", ""),
                    snippet=snippet,
                    content=paper.get("abstract", ""),
                    score=0.8,  # 기본 관련도 점수
                ))
            
            return results
            
        except Exception as e:
            print(f"Semantic Scholar 검색 오류: {e}")
            return []


class CrossRefTool(BaseSearchTool):
    """CrossRef API를 사용한 DOI 검색 도구"""
    
    BASE_URL = "https://api.crossref.org/works"
    
    def __init__(self, mailto: Optional[str] = None):
        self.mailto = mailto or os.getenv("CROSSREF_MAILTO", "")
    
    def is_available(self) -> bool:
        """CrossRef는 항상 사용 가능"""
        return True
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        CrossRef에서 논문을 검색합니다.
        
        Args:
            query: 검색 쿼리 (제목 또는 DOI)
            max_results: 최대 결과 수
            
        Returns:
            SearchResult 리스트
        """
        try:
            params = {
                "query": query,
                "rows": max_results,
            }
            
            headers = {}
            if self.mailto:
                headers["User-Agent"] = f"ReferenceValidator/1.0 (mailto:{self.mailto})"
            
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("message", {}).get("items", []):
                # 제목 추출
                title = item.get("title", [""])[0] if item.get("title") else ""
                
                # 저자 문자열 생성
                authors = []
                for author in item.get("author", [])[:3]:
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                author_str = ", ".join(authors)
                if len(item.get("author", [])) > 3:
                    author_str += " et al."
                
                # 연도 추출
                year = ""
                if "published-print" in item:
                    date_parts = item["published-print"].get("date-parts", [[None]])
                    year = str(date_parts[0][0]) if date_parts[0][0] else ""
                elif "published-online" in item:
                    date_parts = item["published-online"].get("date-parts", [[None]])
                    year = str(date_parts[0][0]) if date_parts[0][0] else ""
                
                # DOI URL
                doi = item.get("DOI", "")
                url = f"https://doi.org/{doi}" if doi else ""
                
                # 스니펫 생성
                snippet = f"[{year}] {author_str}. DOI: {doi}" if doi else f"[{year}] {author_str}"
                
                results.append(SearchResult(
                    source="crossref",
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=None,
                    score=item.get("score", 0.0) / 100.0,  # 정규화
                ))
            
            return results
            
        except Exception as e:
            print(f"CrossRef 검색 오류: {e}")
            return []
    
    def search_by_doi(self, doi: str) -> Optional[SearchResult]:
        """
        DOI로 직접 논문 정보를 조회합니다.
        
        Args:
            doi: DOI 문자열
            
        Returns:
            SearchResult 또는 None
        """
        try:
            # DOI 정규화
            doi = doi.strip()
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            elif doi.startswith("http://doi.org/"):
                doi = doi.replace("http://doi.org/", "")
            
            url = f"{self.BASE_URL}/{doi}"
            
            headers = {}
            if self.mailto:
                headers["User-Agent"] = f"ReferenceValidator/1.0 (mailto:{self.mailto})"
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            item = response.json().get("message", {})
            
            # 제목 추출
            title = item.get("title", [""])[0] if item.get("title") else ""
            
            # 저자 문자열 생성
            authors = []
            for author in item.get("author", [])[:3]:
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                if name:
                    authors.append(name)
            author_str = ", ".join(authors)
            
            return SearchResult(
                source="crossref_doi",
                title=title,
                url=f"https://doi.org/{doi}",
                snippet=f"[DOI] {author_str}",
                content=None,
                score=1.0,  # DOI 직접 조회는 완전 일치
            )
            
        except Exception as e:
            print(f"DOI 조회 오류 ({doi}): {e}")
            return None


# ============================================================================
# 통합 검색 도구
# ============================================================================

class UnifiedSearchTool:
    """모든 검색 도구를 통합하여 관리하는 클래스"""
    
    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        crossref_mailto: Optional[str] = None,
    ):
        self.tavily = TavilySearchTool(api_key=tavily_api_key)
        self.crawler = CrawlerTool()
        self.semantic_scholar = SemanticScholarTool(api_key=semantic_scholar_api_key)
        self.crossref = CrossRefTool(mailto=crossref_mailto)
    
    def search_all(
        self,
        query: str,
        max_results_per_source: int = 3,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        여러 소스에서 동시에 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            max_results_per_source: 소스당 최대 결과 수
            sources: 사용할 소스 목록 (None이면 모두 사용)
            
        Returns:
            모든 소스의 SearchResult 리스트
        """
        all_results = []
        
        available_sources = {
            "tavily": self.tavily,
            "semantic_scholar": self.semantic_scholar,
            "crossref": self.crossref,
        }
        
        if sources is None:
            sources = list(available_sources.keys())
        
        for source_name in sources:
            if source_name in available_sources:
                tool = available_sources[source_name]
                if tool.is_available():
                    try:
                        results = tool.search(query, max_results_per_source)
                        all_results.extend(results)
                        print(f"  [{source_name}] {len(results)}개 결과 검색됨")
                    except Exception as e:
                        print(f"  [{source_name}] 검색 오류: {e}")
                else:
                    print(f"  [{source_name}] 사용 불가능")
        
        return all_results
    
    def search_with_strategies(
        self,
        queries: List[str],
        max_total_results: int = 10,
    ) -> List[SearchResult]:
        """
        여러 검색어로 순차적으로 검색하여 결과를 수집합니다.
        
        Args:
            queries: 검색어 리스트 (우선순위 순)
            max_total_results: 최대 총 결과 수
            
        Returns:
            SearchResult 리스트
        """
        all_results = []
        seen_urls = set()
        
        for query in queries:
            if len(all_results) >= max_total_results:
                break
            
            print(f"\n검색어: {query}")
            results = self.search_all(
                query,
                max_results_per_source=3,
            )
            
            # 중복 제거
            for result in results:
                if result.url not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(result.url)
                
                if len(all_results) >= max_total_results:
                    break
        
        return all_results
    
    def crawl_url(self, url: str) -> Optional[SearchResult]:
        """URL을 직접 크롤링합니다."""
        return self.crawler.crawl_url(url)
    
    def search_by_doi(self, doi: str) -> Optional[SearchResult]:
        """DOI로 논문을 조회합니다."""
        return self.crossref.search_by_doi(doi)


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_search_tool(
    tavily_api_key: Optional[str] = None,
    semantic_scholar_api_key: Optional[str] = None,
    crossref_mailto: Optional[str] = None,
) -> UnifiedSearchTool:
    """
    통합 검색 도구를 생성합니다.
    
    Args:
        tavily_api_key: Tavily API 키
        semantic_scholar_api_key: Semantic Scholar API 키
        crossref_mailto: CrossRef mailto
        
    Returns:
        UnifiedSearchTool 인스턴스
    """
    return UnifiedSearchTool(
        tavily_api_key=tavily_api_key,
        semantic_scholar_api_key=semantic_scholar_api_key,
        crossref_mailto=crossref_mailto,
    )


if __name__ == "__main__":
    # 테스트
    print("=== 검색 도구 테스트 ===")
    
    # 통합 검색 도구 생성
    search_tool = create_search_tool()
    
    # Semantic Scholar 테스트 (API 키 불필요)
    print("\n1. Semantic Scholar 검색:")
    results = search_tool.semantic_scholar.search("attention is all you need transformer", max_results=3)
    for r in results:
        print(f"   - {r.title[:50]}...")
    
    # CrossRef 테스트
    print("\n2. CrossRef 검색:")
    results = search_tool.crossref.search("attention is all you need", max_results=3)
    for r in results:
        print(f"   - {r.title[:50]}...")
    
    # Tavily 테스트 (API 키 필요)
    if search_tool.tavily.is_available():
        print("\n3. Tavily 검색:")
        results = search_tool.tavily.search("attention is all you need", max_results=3)
        for r in results:
            print(f"   - {r.title[:50]}...")
    else:
        print("\n3. Tavily: API 키가 설정되지 않음")
