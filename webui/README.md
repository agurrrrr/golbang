# golbang-webui

`golbang-server`가 `GET /`로 내보내는 최소 내장 UI입니다. 기능은 세 가지뿐입니다.

1. **API 키 설정** — 브라우저 `localStorage`에만 저장하고 요청 헤더로 전달합니다.
2. **추론 속도 통계** — `/metrics`를 폴링해 prefill/decode tok/s, 캐시 재사용률, 대기열,
   슬롯 점유, 503 등을 표시합니다.
3. **스모크 채팅** — 요청을 실제로 만들어 통계를 움직여 보고, 응답마다 그 요청의
   `timings`(prompt/predicted tok/s, 캐시, 드래프트 수락)를 보여 줍니다.

## 빌드

Svelte + Vite로 만들고 `vite-plugin-singlefile`로 **단일 자립형 `dist/index.html`** 을
출력합니다. 이 파일은 리포지토리에 커밋되며 Rust가 `include_bytes!`로 바이너리에
넣습니다. 따라서 `cargo build`에는 Node가 필요하지 않습니다.

```sh
cd webui
npm install
npm run build   # -> webui/dist/index.html
```

UI를 수정한 뒤에는 반드시 `npm run build`를 다시 실행하고 `dist/index.html`을 함께
커밋하십시오. 그래야 바이너리에 반영됩니다.

## 개발

백엔드(`golbang-server`)를 띄운 뒤 `npm run dev`로 Vite 개발 서버를 쓰면 됩니다.
`vite.config.js`에 프록시가 없으므로 필요하면 `/v1`, `/metrics`를 백엔드 주소로
프록시하도록 추가하십시오.
