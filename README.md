# Amazon Bedrock Nova Gallery

Amazon Bedrock의 생성형 AI 모델 Nova를 활용하여 이미지와 비디오를 생성하고 전시하는 애플리케이션입니다.

## 🌟 Features

- **이미지 생성**: Nova Canvas와 Titan image generation G1 v2 모델을 활용한 이미지 생성
- **비디오 생성**: Nova Reel을 활용한 비디오 생성 (24FPS, 6초, 1280x720 해상도)
- **갤러리 뷰**: 생성된 이미지와 비디오의 체계적인 전시
- **히스토리 추적**: 생성 요청 기록과 결과물 조회
- **LLM 기반 프롬프트**: Claude 모델을 활용한 프롬프트 개선

## 📱 사용자 인터페이스

애플리케이션은 4개의 주요 탭으로 구성되어 있습니다.

### 1. 이미지 생성기 - `Nova Canvas`, `Titan Imag Generator v2`

- LLM 기반 프롬프트 개선 (텍스트, 이미지)
  - `Basic Prompt`: 입력한 prompt를 영문으로 변경
  - `Augmented Prompt`: 입력한 prompt와 이미지를 기반으로 프롬프트 작성
- 실시간 이미지 생성 및 미리보기
- Nova Canvas와 Titan 모델 선택 가능

![image-gen-1](./assets/image-gen-1.png)
![image-gen-2](./assets/image-gen-2.png)

### 2. 비디오 생성기 - `Nova Reel`

- LLM 기반 프롬프트 개선 (텍스트, 이미지)
  - `Basic Prompt`: 입력한 prompt를 영문으로 변경
  - `Augmented Prompt`: 입력한 prompt와 이미지를 기반으로 프롬프트 작성
- 텍스트 기반 비디오 생성
- (이미지 + 텍스트) 기반 비디오 생성
  - Image-to-Video의 입력 이미지는 1280x720로 자동 resize (이미지 중앙 기준)
- 비동기 생성 프로세스
- 고정 해상도(1280x720) 및 길이(6초) 지원

![video-gen](./assets/video-gen.png)

### 3. 갤러리

- 생성된 모든 이미지와 비디오 전시
- CloudFront를 통한 최적화된 미디어 전송

![gallery](./assets/gallery.png)

### 4. 히스토리

- 생성 요청 기록 조회
- 상세 옵션 및 결과물 확인

![history](./assets/history.png)

## 🚀 How to Run

### 1. AWS 리소스 준비

- Amazon Bedrock 모델 접근 권한
  - Claude 3.5 (Haiku, Sonnet)
  - Nova Canvas
  - Nova Reel
  - Titan image generation G1 v2
- S3 버킷
- CloudFront 배포
- DynamoDB 테이블

### 2. 환경 변수 설정

**방법 1: 로컬 개발 환경**

- `.env.example`을 `.env`로 복사 후 설정:
  ```sh
  BEDROCK_REGION=us-east-1
  DYNAMO_TABLE=nova-gallery
  S3_BUCKET=nova-gallery-bucket
  CF_DOMAIN=https://abcdefg.cloudfront.net
  ```

**방법 2: AWS Secrets Manager 사용**

- `config.py`에서 `SECRET_NAME` 설정

### 3. 애플리케이션 실행

```sh
streamlit run app.py
```

## 🔄 Architecture / Workflow

### 시스템 아키텍처

```mermaid
graph TB
    App[Streamlit App]
    subgraph "AWS Cloud"
        subgraph "Amazon Bedrock"
            Claude["Claude 3.5<br/>(Haiku/Sonnet)"]
            NovaCanvas[Nova Canvas]
            NovaReel[Nova Reel]
            Titan[Titan Image G1 v2]
        end
        subgraph "Storage & Distribution"
            S3[(S3 Bucket)]
            CF[CloudFront]
            DDB[(DynamoDB)]
        end
    end
    App --> Claude
    App --> NovaCanvas
    App --> NovaReel
    App --> Titan
    NovaCanvas --> S3
    NovaReel --> S3
    Titan --> S3
    App --> DDB
    S3 --> CF
    CF --> App
    DDB --> App
    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px;
    classDef app fill:#85B3D1,stroke:#232F3E,stroke-width:2px;
    classDef bedrock fill:#2E8B57,stroke:#232F3E,stroke-width:2px;
    class S3,CF,DDB aws;
    class App app;
    class Claude,NovaCanvas,NovaReel,Titan bedrock;
```

### 이미지/비디오 생성 Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant Bedrock
    participant DynamoDB
    participant S3

    User->>Streamlit: 1. 이미지/비디오 생성 요청
    Streamlit->>Bedrock: 2. 생성 API 호출

    alt 이미지 생성 (동기)
        Bedrock-->>Streamlit: 3. 이미지 반환
        Streamlit->>S3: 4. 이미지 저장
        Streamlit->>DynamoDB: 5. 요청 정보 저장
    else 비디오 생성 (비동기)
        Streamlit->>DynamoDB: 3. 요청 정보 저장
        Bedrock-->>S3: 4. 비디오 저장
    end
```

### 갤러리/히스토리 조회 Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant Bedrock
    participant DynamoDB
    participant S3
    participant CloudFront

    User->>Streamlit: 1. 갤러리/히스토리 조회
    Streamlit->>Bedrock: 2. 비디오 job 상태 확인
    Bedrock-->>Streamlit: 3. job 상태 반환
    Streamlit->>DynamoDB: 4. 메타데이터 조회
    DynamoDB-->>Streamlit: 5. 메타데이터 반환
    Note over Streamlit: 6. job 상태와 메타데이터 비교
    Streamlit->>DynamoDB: 7. 상태 정보 업데이트
    Streamlit->>CloudFront: 8. 미디어 파일 요청
    CloudFront->>S3: 9. 미디어 파일 조회
    CloudFront-->>Streamlit: 10. 미디어 파일 전송
```

## ⚠️ 주의사항

- CDK 배포 후 CloudFront OAC(Origin Access Control) 설정 확인 필요
- 필요한 모든 AWS 서비스에 대한 적절한 IAM 권한 확인
