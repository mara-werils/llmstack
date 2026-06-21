// Webview script for the LLMStack chat sidebar. Runs in an isolated context and
// talks to the extension host over postMessage. No external dependencies.
(function () {
  const vscode = acquireVsCodeApi();
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");

  let streaming = false;
  let assistantBody = null;

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
    inputEl.value = "";
    assistantBody = addMessage("assistant");
    setBusy(true);
    vscode.postMessage({ type: "send", text: text });
    persist();
  }

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

  window.addEventListener("message", function (event) {
    const msg = event.data;
    if (msg.type === "token") {
      if (!assistantBody) {
        assistantBody = addMessage("assistant");
      }
      assistantBody.textContent += msg.text;
      scrollDown();
      persist();
    } else if (msg.type === "done") {
      setBusy(false);
      assistantBody = null;
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
      persist();
    } else if (msg.type === "clear") {
      showEmpty();
      assistantBody = null;
      setBusy(false);
      persist();
    }
  });

  vscode.postMessage({ type: "ready" });
})();
