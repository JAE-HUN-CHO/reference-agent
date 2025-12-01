"""
LLM 설정 모듈 - Ollama 및 Cloud 모델 지원

지원하는 모델:
- Ollama: llama3, llama3.1, llama3.2, qwen2.5, gemma2, mistral 등
- Cloud: OpenAI (gpt-4, gpt-4o), Anthropic (claude-3), Google (gemini-pro)
"""

import os
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel


class LLMProvider(Enum):
    """LLM 제공자 열거형"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class LLMConfig:
    """LLM 설정 데이터 클래스"""
    provider: LLMProvider
    model_name: str
    temperature: float = 0.1
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # Ollama 기본 설정
    ollama_base_url: str = "http://localhost:11434"
    
    # 재시도 설정
    max_retries: int = 3
    retry_delay: float = 1.0


class LLMFactory:
    """LLM 인스턴스 생성 팩토리"""
    
    @staticmethod
    def create_llm(config: LLMConfig) -> BaseChatModel:
        """
        설정에 따라 적절한 LLM 인스턴스를 생성합니다.
        
        Args:
            config: LLM 설정
            
        Returns:
            LangChain 호환 LLM 인스턴스
        """
        if config.provider == LLMProvider.OLLAMA:
            return LLMFactory._create_ollama_llm(config)
        elif config.provider == LLMProvider.OPENAI:
            return LLMFactory._create_openai_llm(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return LLMFactory._create_anthropic_llm(config)
        elif config.provider == LLMProvider.GOOGLE:
            return LLMFactory._create_google_llm(config)
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {config.provider}")
    
    @staticmethod
    def _create_ollama_llm(config: LLMConfig) -> BaseChatModel:
        """Ollama LLM 인스턴스 생성"""
        from langchain_ollama import ChatOllama
        
        return ChatOllama(
            model=config.model_name,
            base_url=config.base_url or config.ollama_base_url,
            temperature=config.temperature,
        )
    
    @staticmethod
    def _create_openai_llm(config: LLMConfig) -> BaseChatModel:
        """OpenAI LLM 인스턴스 생성"""
        from langchain_openai import ChatOpenAI
        
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. OPENAI_API_KEY 환경 변수를 설정하세요.")
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url=config.base_url,
        )
    
    @staticmethod
    def _create_anthropic_llm(config: LLMConfig) -> BaseChatModel:
        """Anthropic LLM 인스턴스 생성"""
        from langchain_anthropic import ChatAnthropic
        
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API 키가 필요합니다. ANTHROPIC_API_KEY 환경 변수를 설정하세요.")
        
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )
    
    @staticmethod
    def _create_google_llm(config: LLMConfig) -> BaseChatModel:
        """Google LLM 인스턴스 생성"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API 키가 필요합니다. GOOGLE_API_KEY 환경 변수를 설정하세요.")
        
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=api_key,
        )


# 기본 설정 프리셋
class ModelPresets:
    """모델 프리셋 설정"""
    
    # Ollama 프리셋
    OLLAMA_GLM4_CLOUD = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="glm-4.6:cloud",
        temperature=0.1,
    )
    
    OLLAMA_LLAMA3_2 = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.2",
        temperature=0.1,
    )
    
    OLLAMA_QWEN2_5 = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="qwen2.5",
        temperature=0.1,
    )
    
    OLLAMA_GEMMA2 = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="gemma2",
        temperature=0.1,
    )
    
    OLLAMA_MISTRAL = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="mistral",
        temperature=0.1,
    )
    
    # OpenAI 프리셋
    OPENAI_GPT4O = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        temperature=0.1,
    )
    
    OPENAI_GPT4O_MINI = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        temperature=0.1,
    )
    
    # Anthropic 프리셋
    ANTHROPIC_CLAUDE_SONNET = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
        temperature=0.1,
    )
    
    ANTHROPIC_CLAUDE_HAIKU = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.1,
    )
    
    # Google 프리셋
    GOOGLE_GEMINI_PRO = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-2.5-pro",
        temperature=0.1,
    )
    
    GOOGLE_GEMINI_FLASH = LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-2.5-flash",
        temperature=0.1,
    )


class Settings:
    """전역 설정 클래스"""
    
    def __init__(
        self,
        # LLM 설정 (용도별 모델 지정)
        parser_llm_config: Optional[LLMConfig] = None,
        web_llm_config: Optional[LLMConfig] = None,
        validation_llm_config: Optional[LLMConfig] = None,
        context_llm_config: Optional[LLMConfig] = None,
        citation_llm_config: Optional[LLMConfig] = None,
        report_llm_config: Optional[LLMConfig] = None,
        
        # 검색 설정
        tavily_api_key: Optional[str] = None,
        
        # 검증 설정
        max_retry_attempts: int = 3,
        title_similarity_threshold: float = 0.85,
        author_match_threshold: float = 0.5,
        
        # 경로 설정
        workspace_dir: str = "workspace",
    ):
        # 기본 LLM으로 Ollama glm-4.6:cloud 사용
        default_config = ModelPresets.OLLAMA_GLM4_CLOUD
        
        self.parser_llm_config = parser_llm_config or default_config
        self.web_llm_config = web_llm_config or default_config
        self.validation_llm_config = validation_llm_config or default_config
        self.context_llm_config = context_llm_config or default_config
        self.citation_llm_config = citation_llm_config or default_config
        self.report_llm_config = report_llm_config or default_config
        
        # Tavily API 키
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        # 검증 설정
        self.max_retry_attempts = max_retry_attempts
        self.title_similarity_threshold = title_similarity_threshold
        self.author_match_threshold = author_match_threshold
        
        # 경로 설정
        self.workspace_dir = workspace_dir
    
    def get_parser_llm(self) -> BaseChatModel:
        """Parser Agent용 LLM 반환"""
        return LLMFactory.create_llm(self.parser_llm_config)
    
    def get_web_llm(self) -> BaseChatModel:
        """Web Agent용 LLM 반환"""
        return LLMFactory.create_llm(self.web_llm_config)
    
    def get_validation_llm(self) -> BaseChatModel:
        """Validation Agent용 LLM 반환"""
        return LLMFactory.create_llm(self.validation_llm_config)
    
    def get_context_llm(self) -> BaseChatModel:
        """Context Agent용 LLM 반환"""
        return LLMFactory.create_llm(self.context_llm_config)
    
    def get_citation_llm(self) -> BaseChatModel:
        """Citation Agent용 LLM 반환"""
        return LLMFactory.create_llm(self.citation_llm_config)
    
    def get_report_llm(self) -> BaseChatModel:
        """Report Agent용 LLM 반환"""
        return LLMFactory.create_llm(self.report_llm_config)


# 전역 설정 인스턴스 (기본값)
settings = Settings()


def configure(
    provider: str = "ollama",
    model_name: str = "glm-4.6:cloud",
    temperature: float = 0.1,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    **kwargs
) -> Settings:
    """
    전역 설정을 간편하게 구성합니다.
    
    Args:
        provider: LLM 제공자 ("ollama", "openai", "anthropic", "google")
        model_name: 모델 이름
        temperature: 생성 온도
        base_url: 커스텀 API URL (Ollama 등)
        api_key: API 키
        tavily_api_key: Tavily API 키
        **kwargs: 추가 설정
        
    Returns:
        Settings 인스턴스
    """
    global settings
    
    # Provider 매핑
    provider_map = {
        "ollama": LLMProvider.OLLAMA,
        "openai": LLMProvider.OPENAI,
        "anthropic": LLMProvider.ANTHROPIC,
        "google": LLMProvider.GOOGLE,
    }
    
    llm_provider = provider_map.get(provider.lower())
    if not llm_provider:
        raise ValueError(f"지원하지 않는 제공자: {provider}")
    
    # 단일 LLM 설정 생성
    llm_config = LLMConfig(
        provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    
    # 모든 에이전트에 동일한 설정 적용
    settings = Settings(
        parser_llm_config=llm_config,
        web_llm_config=llm_config,
        validation_llm_config=llm_config,
        context_llm_config=llm_config,
        citation_llm_config=llm_config,
        report_llm_config=llm_config,
        tavily_api_key=tavily_api_key,
        **kwargs
    )
    
    return settings


def configure_multi_model(
    parser_config: Optional[LLMConfig] = None,
    web_config: Optional[LLMConfig] = None,
    validation_config: Optional[LLMConfig] = None,
    context_config: Optional[LLMConfig] = None,
    citation_config: Optional[LLMConfig] = None,
    report_config: Optional[LLMConfig] = None,
    tavily_api_key: Optional[str] = None,
    **kwargs
) -> Settings:
    """
    에이전트별로 다른 LLM을 구성합니다.
    
    예시:
        configure_multi_model(
            parser_config=ModelPresets.OLLAMA_LLAMA3_2,
            validation_config=ModelPresets.OPENAI_GPT4O,
            context_config=ModelPresets.GOOGLE_GEMINI_PRO,
        )
    """
    global settings
    
    settings = Settings(
        parser_llm_config=parser_config,
        web_llm_config=web_config,
        validation_llm_config=validation_config,
        context_llm_config=context_config,
        citation_llm_config=citation_config,
        report_llm_config=report_config,
        tavily_api_key=tavily_api_key,
        **kwargs
    )
    
    return settings


if __name__ == "__main__":
    # 테스트
    print("=== LLM 설정 테스트 ===")
    
    # Ollama 설정 테스트
    print("\n1. Ollama 설정:")
    configure(provider="ollama", model_name="llama3.2")
    print(f"   Provider: {settings.parser_llm_config.provider}")
    print(f"   Model: {settings.parser_llm_config.model_name}")
    
    # 멀티 모델 설정 테스트
    print("\n2. 멀티 모델 설정:")
    configure_multi_model(
        parser_config=ModelPresets.OLLAMA_LLAMA3_2,
        validation_config=ModelPresets.OPENAI_GPT4O,
    )
    print(f"   Parser: {settings.parser_llm_config.model_name}")
    print(f"   Validation: {settings.validation_llm_config.model_name}")
