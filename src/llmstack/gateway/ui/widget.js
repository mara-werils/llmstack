// LLMStack Embeddable Chat Widget — zero-dependency, shadow DOM isolated
// Usage: <script src="/ui/widget.js" data-api-url="http://localhost:8000" data-model="auto"></script>
(function () {
  "use strict";
  const scriptTag = document.currentScript;
  const API_URL = (scriptTag && scriptTag.getAttribute("data-api-url")) || window.location.origin;
  const MODEL = (scriptTag && scriptTag.getAttribute("data-model")) || "auto";

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const T = prefersDark
    ? { bg: "#1e1e2e", fg: "#cdd6f4", input: "#313244", border: "#45475a", accent: "#89b4fa", user: "#585b70", ai: "#313244", hover: "#74c7ec" }
    : { bg: "#ffffff", fg: "#1e1e2e", input: "#f5f5f5", border: "#e0e0e0", accent: "#1e66f5", user: "#dce5ff", ai: "#f0f0f0", hover: "#04a5e5" };

  const host = document.createElement("div");
  document.body.appendChild(host);
  const shadow = host.attachShadow({ mode: "closed" });

  const style = document.createElement("style");
  style.textContent = `
    *{box-sizing:border-box;margin:0;padding:0}
    .llm-btn{position:fixed;bottom:20px;right:20px;width:56px;height:56px;border-radius:50%;
      background:${T.accent};color:#fff;border:none;cursor:pointer;font-size:24px;
      box-shadow:0 4px 12px rgba(0,0,0,.25);z-index:999999;display:flex;align-items:center;
      justify-content:center;transition:background .2s}
    .llm-btn:hover{background:${T.hover}}
    .llm-win{position:fixed;bottom:88px;right:20px;width:380px;height:520px;
      background:${T.bg};border:1px solid ${T.border};border-radius:12px;
      box-shadow:0 8px 32px rgba(0,0,0,.2);z-index:999999;display:none;
      flex-direction:column;font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
      color:${T.fg};overflow:hidden}
    .llm-win.open{display:flex}
    .llm-hdr{padding:14px 16px;background:${T.accent};color:#fff;font-weight:600;
      font-size:15px;display:flex;align-items:center;justify-content:space-between}
    .llm-hdr button{background:none;border:none;color:#fff;cursor:pointer;font-size:18px;
      line-height:1;opacity:.8}
    .llm-hdr button:hover{opacity:1}
    .llm-msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
    .llm-msg{max-width:85%;padding:10px 14px;border-radius:12px;font-size:14px;
      line-height:1.5;word-wrap:break-word;white-space:pre-wrap}
    .llm-msg.user{background:${T.user};align-self:flex-end;border-bottom-right-radius:4px}
    .llm-msg.ai{background:${T.ai};align-self:flex-start;border-bottom-left-radius:4px}
    .llm-typing{align-self:flex-start;padding:10px 14px;font-size:14px;color:${T.fg};opacity:.6}
    .llm-typing span{animation:blink 1.4s infinite both}
    .llm-typing span:nth-child(2){animation-delay:.2s}
    .llm-typing span:nth-child(3){animation-delay:.4s}
    @keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
    .llm-bar{padding:10px 12px;border-top:1px solid ${T.border};display:flex;gap:8px}
    .llm-bar textarea{flex:1;resize:none;border:1px solid ${T.border};border-radius:8px;
      padding:8px 12px;font-size:14px;font-family:inherit;background:${T.input};
      color:${T.fg};outline:none;min-height:40px;max-height:100px}
    .llm-bar textarea:focus{border-color:${T.accent}}
    .llm-bar button{background:${T.accent};color:#fff;border:none;border-radius:8px;
      padding:0 16px;cursor:pointer;font-size:14px;font-weight:600;transition:background .2s}
    .llm-bar button:hover{background:${T.hover}}
    .llm-bar button:disabled{opacity:.5;cursor:not-allowed}
    @media(max-width:480px){
      .llm-win{bottom:0;right:0;left:0;width:100%;height:100%;border-radius:0}
      .llm-btn{bottom:12px;right:12px;width:48px;height:48px;font-size:20px}
    }
  `;
  shadow.appendChild(style);

  // Chat button
  const btn = document.createElement("button");
  btn.className = "llm-btn";
  btn.innerHTML = "&#x1F4AC;";
  btn.setAttribute("aria-label", "Open chat");
  shadow.appendChild(btn);

  // Chat window
  const win = document.createElement("div");
  win.className = "llm-win";
  win.innerHTML = `
    <div class="llm-hdr"><span>LLMStack Chat</span><button aria-label="Close">&times;</button></div>
    <div class="llm-msgs"></div>
    <div class="llm-bar">
      <textarea rows="1" placeholder="Type a message..." aria-label="Message"></textarea>
      <button>Send</button>
    </div>`;
  shadow.appendChild(win);

  const msgs = win.querySelector(".llm-msgs");
  const input = win.querySelector("textarea");
  const sendBtn = win.querySelector(".llm-bar button");
  const closeBtn = win.querySelector(".llm-hdr button");

  let history = [];
  let streaming = false;

  btn.onclick = () => { win.classList.toggle("open"); if (win.classList.contains("open")) input.focus(); };
  closeBtn.onclick = () => win.classList.remove("open");

  function addMsg(role, text) {
    const el = document.createElement("div");
    el.className = "llm-msg " + (role === "user" ? "user" : "ai");
    el.textContent = text;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "llm-typing";
    el.innerHTML = "<span>.</span><span>.</span><span>.</span>";
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  async function send() {
    const text = input.value.trim();
    if (!text || streaming) return;
    input.value = "";
    input.style.height = "auto";
    addMsg("user", text);
    history.push({ role: "user", content: text });

    streaming = true;
    sendBtn.disabled = true;
    const typing = showTyping();
    let aiEl = null;
    let full = "";

    try {
      const res = await fetch(API_URL + "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: MODEL, messages: history, stream: true }),
      });

      if (!res.ok) {
        typing.remove();
        addMsg("ai", "Error: " + res.status + " " + res.statusText);
        streaming = false;
        sendBtn.disabled = false;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          const data = trimmed.slice(6);
          if (data === "[DONE]") break;
          try {
            const json = JSON.parse(data);
            const delta = json.choices && json.choices[0] && json.choices[0].delta;
            if (delta && delta.content) {
              if (!aiEl) { typing.remove(); aiEl = addMsg("ai", ""); }
              full += delta.content;
              aiEl.textContent = full;
              msgs.scrollTop = msgs.scrollHeight;
            }
          } catch (_) { /* skip malformed chunks */ }
        }
      }

      if (!aiEl) { typing.remove(); aiEl = addMsg("ai", full || "(empty response)"); }
      history.push({ role: "assistant", content: full });
    } catch (err) {
      typing.remove();
      addMsg("ai", "Connection error: " + err.message);
    }

    streaming = false;
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.onclick = send;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 100) + "px";
  });
})();
