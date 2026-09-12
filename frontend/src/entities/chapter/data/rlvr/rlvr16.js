// rlvr/rlvr16.js
export default {
  "id": "rlvr16",
  "num": "16",
  "title": "Chapter 16: 大模型推論極限優化 — 量化 (GPTQ/AWQ)、投機解碼與硬體編譯",
  "icon": "🚀",
  "file": "16_inference_optimization_quantization_compilation",
  "hasVisualizer": "rlvr_scaling",
  "readTime": "26 min",
  "summary": "Roofline 模型、GPTQ/AWQ 4-bit 量化、投機解碼無損驗證定理、PagedAttention 與 Apple MLX 編譯",
  "toc": [
    {
      "level": 2,
      "text": "16.1 推論雙階段瓶頸：Roofline 模型與計算/帶寬邊界",
      "anchor": "161-推論雙階段瓶頸roofline-模型與計算帶寬邊界"
    },
    {
      "level": 2,
      "text": "16.2 權重量化技術：GPTQ vs AWQ",
      "anchor": "162-權重量化技術gptq-vs-awq"
    },
    {
      "level": 2,
      "text": "16.3 投機解碼 (Speculative Decoding)：無損加速之美",
      "anchor": "163-投機解碼-speculative-decoding無損加速之美"
    },
    {
      "level": 2,
      "text": "16.4 KV-Cache 極限優化：PagedAttention 與 MLA",
      "anchor": "164-kv-cache-極限優化pagedattention-與-mla"
    },
    {
      "level": 2,
      "text": "16.5 硬體感知編譯：torch.compile 與 Apple MLX",
      "anchor": "165-硬體感知編譯torchcompile-與-apple-mlx"
    },
    {
      "level": 2,
      "text": "16.6 面試高頻考點與系統設計 (Interview Playbook)",
      "anchor": "166-面試高頻考點與系統設計-interview-playbook"
    }
  ],
  "codeLines": []
};
