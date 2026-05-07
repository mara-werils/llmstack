# VHS tape for recording llmstack demo GIF
# Install VHS: https://github.com/charmbracelet/vhs
# Run: vhs demo.tape

Output demo.gif

Set FontSize 16
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20

# --- Scene 1: Show version ---
Type "llmstack --version"
Enter
Sleep 1s

# --- Scene 2: Initialize with RAG preset ---
Type "llmstack init --preset rag"
Enter
Sleep 3s

# --- Scene 3: Show generated config ---
Type "cat llmstack.yaml"
Enter
Sleep 3s

# --- Scene 4: Start the stack ---
Type "llmstack up"
Enter
Sleep 5s

# --- Scene 5: Check status ---
Type "llmstack status"
Enter
Sleep 3s

# --- Scene 6: Send a request ---
Type@50ms "curl -s http://localhost:8000/v1/chat/completions \\"
Enter
Type@50ms '  -H "Authorization: Bearer llmstack-key" \\'
Enter
Type@50ms '  -H "Content-Type: application/json" \\'
Enter
Type@50ms '  -d '\''{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'\'' | jq .'
Enter
Sleep 4s

# --- Scene 7: Tear down ---
Type "llmstack down"
Enter
Sleep 3s
