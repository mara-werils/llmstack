// Webview script for the LLMStack chat sidebar. Runs in an isolated context and
// talks to the extension host over postMessage. No external dependencies.
(function () {
  const vscode = acquireVsCodeApi();
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const ctxToggle = document.getElementById("ctx-toggle");
  const ctxLabel = document.getElementById("ctx-label");
  const modelSelect = document.getElementById("model");
  const banner = document.getElementById("banner");
  const bannerStart = document.getElementById("banner-start");
  const bannerHelp = document.getElementById("banner-help");

  let streaming = false;
  let assistantBody = null;
  let assistantRaw = "";
  let lastUserText = "";

  const saved = vscode.getState();
  if (saved && saved.html) {
    messagesEl.innerHTML = saved.html;
  } else {
    showEmpty();
  }

  function showEmpty() {
    messagesEl.innerHTML =
      '<div class="empty">Ask your local model anything.<br/>Code never leaves your machine.</div>';
  }

  function persist() {
    vscode.setState({ html: messagesEl.innerHTML });
  }

  function clearEmpty() {
    const e = messagesEl.querySelector(".empty");
    if (e) {
      e.remove();
    }
  }

  function scrollDown() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addMessage(role) {
    clearEmpty();
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;

    const label = document.createElement("div");
    label.className = "role";
    label.textContent = role === "user" ? "You" : "LLMStack";

    const body = document.createElement("div");
    body.className = "body";

    wrap.appendChild(label);
    wrap.appendChild(body);
    messagesEl.appendChild(wrap);
    scrollDown();
    return body;
  }

  function setBusy(value) {
    streaming = value;
    sendBtn.textContent = value ? "Stop" : "Send";
    sendBtn.classList.toggle("stop", value);
  }

  function send() {
    const text = inputEl.value.trim();
    if (!text || streaming) {
      return;
    }
    addMessage("user").textContent = text;
    lastUserText = text;
    inputEl.value = "";
    assistantBody = addMessage("assistant");
    assistantRaw = "";
    setBusy(true);
    vscode.postMessage({
      type: "send",
      text: text,
      includeContext: ctxToggle.checked,
    });
    persist();
  }

  // --- Minimal, safe Markdown rendering (code fences + inline code) ----------

  function splitFences(raw) {
    const lines = raw.split("\n");
    const parts = [];
    let inCode = false;
    let lang = "";
    let buf = [];
    function flush(type) {
      const content = buf.join("\n");
      if (type === "code") {
        parts.push({ type: "code", lang: lang, content: content });
      } else if (content.length) {
        parts.push({ type: "text", content: content });
      }
      buf = [];
    }
    for (const line of lines) {
      const m = /^```(.*)$/.exec(line);
      if (m) {
        if (!inCode) {
          flush("text");
          inCode = true;
          lang = m[1].trim();
        } else {
          flush("code");
          inCode = false;
          lang = "";
        }
      } else {
        buf.push(line);
      }
    }
    flush(inCode ? "code" : "text");
    return parts;
  }

  function makeCodeBlock(lang, content) {
    const wrap = document.createElement("div");
    wrap.className = "code-block";

    const header = document.createElement("div");
    header.className = "code-header";
    const langEl = document.createElement("span");
    langEl.className = "code-lang";
    langEl.textContent = lang || "code";
    const actions = document.createElement("div");
    actions.className = "code-actions";
    actions.appendChild(makeAction("apply", "Apply"));
    actions.appendChild(makeAction("insert", "Insert"));
    actions.appendChild(makeAction("copy", "Copy"));
    header.appendChild(langEl);
    header.appendChild(actions);

    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = content;
    pre.appendChild(code);

    wrap.appendChild(header);
    wrap.appendChild(pre);
    return wrap;
  }

  function makeAction(action, label) {
    const btn = document.createElement("button");
    btn.className = "code-action";
    btn.dataset.action = action;
    btn.textContent = label;
    return btn;
  }

  function makeVote(vote, label) {
    const btn = document.createElement("button");
    btn.className = "fb-btn";
    btn.dataset.vote = vote;
    btn.textContent = label;
    return btn;
  }

  function addFeedback(wrap) {
    if (!wrap || wrap.querySelector(".feedback")) {
      return;
    }
    const fb = document.createElement("div");
    fb.className = "feedback";
    fb.appendChild(makeVote("up", "👍"));
    fb.appendChild(makeVote("down", "👎"));
    wrap.appendChild(fb);
  }

  function appendText(container, text) {
    const div = document.createElement("div");
    div.className = "md-text";
    const tokens = text.split(/(`[^`]+`)/g);
    for (const tok of tokens) {
      if (tok.length >= 2 && tok[0] === "`" && tok[tok.length - 1] === "`") {
        const code = document.createElement("code");
        code.className = "inline";
        code.textContent = tok.slice(1, -1);
        div.appendChild(code);
      } else if (tok) {
        div.appendChild(document.createTextNode(tok));
      }
    }
    container.appendChild(div);
  }

  function renderMarkdown(container, raw) {
    container.textContent = "";
    for (const part of splitFences(raw)) {
      if (part.type === "code") {
        container.appendChild(makeCodeBlock(part.lang, part.content));
      } else {
        appendText(container, part.content);
      }
    }
  }

  // --- Event wiring ----------------------------------------------------------

  sendBtn.addEventListener("click", function () {
    if (streaming) {
      vscode.postMessage({ type: "stop" });
    } else {
      send();
    }
  });

  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  modelSelect.addEventListener("change", function () {
    vscode.postMessage({ type: "model", text: modelSelect.value });
  });

  bannerStart.addEventListener("click", function () {
    vscode.postMessage({ type: "startGateway" });
  });

  bannerHelp.addEventListener("click", function () {
    vscode.postMessage({ type: "openWalkthrough" });
  });

  // Delegated handler so restored buttons keep working after a reload.
  messagesEl.addEventListener("click", function (e) {
    const target = e.target;
    if (!target || !target.closest) {
      return;
    }

    const fbBtn = target.closest(".fb-btn");
    if (fbBtn) {
      const wrap = fbBtn.closest(".msg");
      vscode.postMessage({
        type: "feedback",
        vote: fbBtn.dataset.vote,
        query: (wrap && wrap.dataset.query) || "",
        response: (wrap && wrap.dataset.response) || "",
      });
      const siblings = fbBtn.parentElement.querySelectorAll(".fb-btn");
      siblings.forEach(function (x) {
        x.classList.remove("chosen");
      });
      fbBtn.classList.add("chosen");
      return;
    }

    const btn = target.closest(".code-action");
    if (!btn) {
      return;
    }
    const block = btn.closest(".code-block");
    const codeEl = block && block.querySelector("code");
    const text = codeEl ? codeEl.textContent : "";
    vscode.postMessage({ type: btn.dataset.action, text: text });
  });

  window.addEventListener("message", function (event) {
    const msg = event.data;
    if (msg.type === "token") {
      if (!assistantBody) {
        assistantBody = addMessage("assistant");
        assistantRaw = "";
      }
      assistantRaw += msg.text;
      assistantBody.textContent = assistantRaw;
      scrollDown();
    } else if (msg.type === "done") {
      if (assistantBody) {
        const finalResponse = assistantRaw;
        renderMarkdown(assistantBody, finalResponse);
        const wrap = assistantBody.parentElement;
        if (wrap) {
          wrap.dataset.query = lastUserText;
          wrap.dataset.response = finalResponse;
          addFeedback(wrap);
        }
      }
      setBusy(false);
      assistantBody = null;
      assistantRaw = "";
      scrollDown();
      persist();
    } else if (msg.type === "error") {
      if (!assistantBody) {
        assistantBody = addMessage("assistant");
      }
      assistantBody.classList.add("error");
      assistantBody.textContent +=
        (assistantBody.textContent ? "\n\n" : "") + "⚠ " + msg.message;
      setBusy(false);
      assistantBody = null;
      assistantRaw = "";
      persist();
    } else if (msg.type === "clear") {
      showEmpty();
      assistantBody = null;
      assistantRaw = "";
      setBusy(false);
      persist();
    } else if (msg.type === "health") {
      banner.hidden = !!msg.ok;
    } else if (msg.type === "models") {
      modelSelect.innerHTML = "";
      const models =
        msg.models && msg.models.length
          ? msg.models
          : msg.current
            ? [msg.current]
            : [];
      for (const id of models) {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = id;
        if (id === msg.current) {
          opt.selected = true;
        }
        modelSelect.appendChild(opt);
      }
      modelSelect.disabled = models.length === 0;
    } else if (msg.type === "context") {
      if (msg.file) {
        ctxLabel.textContent =
          (msg.hasSelection ? "Include selection · " : "Include file · ") + msg.file;
        ctxToggle.disabled = false;
      } else {
        ctxLabel.textContent = "Include editor context";
        ctxToggle.checked = false;
        ctxToggle.disabled = true;
      }
    }
  });

  vscode.postMessage({ type: "ready" });
})();
