# VHS tape for recording llmstack demo GIF
# Pre-requisite: cp scripts/fake-llmstack /tmp/llmstack
# Run: vhs demo.tape

Output demo.gif

Set FontSize 16
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 40ms
Set Shell "bash"

# Set PATH so /tmp/llmstack is found, then clear screen
Hide
Type "export PATH=/tmp:$PATH"
Enter
Sleep 300ms
Type "export PS1='$ '"
Enter
Sleep 300ms
Type "clear"
Enter
Sleep 500ms
Show

# --- Scene 1: Show version ---
Type "llmstack --version"
Enter
Sleep 1.5s

# --- Scene 2: Initialize with RAG preset ---
Type "llmstack init --preset rag"
Enter
Sleep 3s

# --- Scene 3: Start the stack ---
Type "llmstack up"
Enter
Sleep 4s

# --- Scene 4: Check status ---
Type "llmstack status"
Enter
Sleep 3s

# --- Scene 5: Tear down ---
Type "llmstack down"
Enter
Sleep 3s
