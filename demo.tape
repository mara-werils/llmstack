Output assets/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 900
Set Height 500
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms
Set Padding 20

# Show install
Type "pip install llmstack-cli" Sleep 500ms Enter
Sleep 2s

# Single question
Type "llmstack ask 'How does authentication work?' ./src/" Sleep 500ms Enter
Sleep 6s

# Interactive mode
Type "llmstack ask -i ./src/" Sleep 500ms Enter
Sleep 3s
Type "How does the cache work?" Sleep 500ms Enter
Sleep 5s
Type "What happens when Redis goes down?" Sleep 500ms Enter
Sleep 5s
Type "exit" Enter
Sleep 1s
