# genai-kit

## Path 설정 관련 안내

모듈을 사용할 때 import 과정에서 path 관련 문제가 발생할 수 있습니다. 특히, 현재 모듈이 다른 디렉토리 또는 상위 디렉토리에 있을 경우, Python이 해당 모듈을 찾지 못할 수 있습니다. 이 문제를 해결하려면 다음과 같은 방법으로 sys.path에 모듈 경로를 추가할 수 있습니다:

```py
import os
import sys
sys.path.append(os.path.abspath("../"))
```

## Subtree

해당 모듈을 사용하기 위해 subtree 형태로 추가할 수 있습니다.

### Subtree 추가

```sh
# 하위 저장소 원격 추가
git remote add genai_kit https://github.com/hi-space/genai-kit.git

# 추가된 원격 확인
git remote -v

# subtree 추가
# --squash: 하위 저장소의 여러 커밋을 하나의 커밋으로 합쳐서 병합 (상위 저장소의 커밋 히스토리를 깔끔하게 유지)
git subtree add --prefix=genai_kit genai_kit main --squash
```

### Subtree Push/Pull

```sh
# git subtree push --prefix=<디렉토리 이름> <원격 이름> <브랜치 이름>
git subtree push --prefix=genai_kit genai_kit main

# git subtree pull --prefix=<디렉토리 이름> <원격 이름> <브랜치 이름>
git subtree pull --prefix=genai_kit genai_kit main --squash
```

### Subtree 제거

```sh
# Subtree로 추가된 디렉토리를 삭제
git rm -r genai_kit

# 원격 연결 삭제
git remote remove genai_kit
```
