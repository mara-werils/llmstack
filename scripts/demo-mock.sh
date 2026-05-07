#!/usr/bin/env bash
# Mock script for demo GIF recording — simulates llmstack CLI output

set -e

CMD="${1:-}"

case "$CMD" in
  version)
    echo "llmstack 0.1.0"
    ;;
  init)
    echo ""
    echo -e "\033[36mHardware detected:\033[0m"
    echo "  CPU: 10 cores"
    echo "  RAM: 32 GB"
    echo "  GPU: Apple M2 Pro (16 GB VRAM)"
    echo ""
    echo -e "\033[36mUsing preset:\033[0m rag"
    echo -e "  Backend: \033[32mOllama\033[0m"
    echo ""
    echo -e "\033[32mCreated llmstack.yaml\033[0m"
    echo "Next: edit the config if needed, then run llmstack up"
    ;;
  up)
    echo ""
    echo -e "\033[1mStarting LLMStack...\033[0m"
    echo ""
    sleep 0.3
    echo -e "  \033[32m✓\033[0m qdrant        running   :6333"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m redis         running   :6379"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m ollama        running   :11434"
    sleep 0.3
    echo -e "  \033[32m✓\033[0m tei           running   :8002"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m gateway       running   :8000"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m prometheus    running   :9090"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m grafana       running   :8080"
    echo ""
    echo -e "\033[32mStack is ready!\033[0m 7 services running."
    echo "API: http://localhost:8000/v1"
    echo "Dashboard: http://localhost:8080"
    ;;
  status)
    echo ""
    echo -e "\033[1m            LLMStack Status\033[0m"
    echo "┌─────────────┬──────────┬─────────┬──────────────┐"
    echo "│ Service     │ Container│ Status  │ Ports        │"
    echo "├─────────────┼──────────┼─────────┼──────────────┤"
    echo -e "│ qdrant      │ a3f1..   │ \033[32mrunning\033[0m │ 6333->6333   │"
    echo -e "│ redis       │ b7e2..   │ \033[32mrunning\033[0m │ 6379->6379   │"
    echo -e "│ ollama      │ c9d4..   │ \033[32mrunning\033[0m │ 11434->11434 │"
    echo -e "│ tei         │ d2a8..   │ \033[32mrunning\033[0m │ 8002->8002   │"
    echo -e "│ gateway     │ e5c1..   │ \033[32mrunning\033[0m │ 8000->8000   │"
    echo -e "│ prometheus  │ f8b3..   │ \033[32mrunning\033[0m │ 9090->9090   │"
    echo -e "│ grafana     │ 1a7e..   │ \033[32mrunning\033[0m │ 8080->8080   │"
    echo "└─────────────┴──────────┴─────────┴──────────────┘"
    ;;
  down)
    echo ""
    echo "Stopping LLMStack..."
    echo ""
    sleep 0.2
    echo -e "  \033[32m✓\033[0m grafana       stopped"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m prometheus    stopped"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m gateway       stopped"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m tei           stopped"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m ollama        stopped"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m redis         stopped"
    sleep 0.2
    echo -e "  \033[32m✓\033[0m qdrant        stopped"
    echo ""
    echo -e "\033[32mAll services stopped.\033[0m"
    ;;
esac
