// 2048 DQN Game JavaScript
class Game2048DQN {
    constructor() {
        this.gameState = null;
        this.autoPlayInterval = null;
        this.currentMode = 'manual';
        
        this.initializeElements();
        this.bindEvents();
        this.loadGameState();
    }

    initializeElements() {
        // Game elements
        this.gameBoard = document.getElementById('game-board');
        this.scoreElement = document.getElementById('score');
        this.maxTileElement = document.getElementById('max-tile');
        this.movesElement = document.getElementById('moves');
        this.epsilonElement = document.getElementById('epsilon');
        this.messagesElement = document.getElementById('messages');
        
        // Control elements
        this.modelPathInput = document.getElementById('model-path');
        this.loadModelBtn = document.getElementById('load-model-btn');
        this.modelStatusElement = document.getElementById('model-status');
        this.resetBtn = document.getElementById('reset-btn');
        this.manualModeBtn = document.getElementById('manual-mode-btn');
        this.autoModeBtn = document.getElementById('auto-mode-btn');
        
        // Auto play controls
        this.autoControls = document.getElementById('auto-controls');
        this.autoStepBtn = document.getElementById('auto-step-btn');
        this.autoPlayBtn = document.getElementById('auto-play-btn');
        this.autoStopBtn = document.getElementById('auto-stop-btn');
        
        // Manual controls
        this.manualControls = document.getElementById('manual-controls');
        this.controlButtons = document.querySelectorAll('.control-btn[data-direction]');
        
        // History
        this.historyContent = document.getElementById('history-content');
        this.clearHistoryBtn = document.getElementById('clear-history-btn');
    }

    bindEvents() {
        // Model controls
        this.loadModelBtn.addEventListener('click', () => this.loadModel());
        this.modelPathInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.loadModel();
        });

        // Game controls
        this.resetBtn.addEventListener('click', () => this.resetGame());
        this.manualModeBtn.addEventListener('click', () => this.setMode('manual'));
        this.autoModeBtn.addEventListener('click', () => this.setMode('auto'));

        // Auto play controls
        this.autoStepBtn.addEventListener('click', () => this.makeAutoMove());
        this.autoPlayBtn.addEventListener('click', () => this.startAutoPlay());
        this.autoStopBtn.addEventListener('click', () => this.stopAutoPlay());

        // Manual controls
        this.controlButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const direction = btn.getAttribute('data-direction');
                this.makeManualMove(direction);
            });
        });

        // Keyboard controls
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // History
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
    }

    async loadGameState() {
        try {
            const response = await fetch('/api/state');
            const data = await response.json();
            this.updateGameDisplay(data);
        } catch (error) {
            this.showMessage('Error loading game state', 'error');
        }
    }

    async loadModel() {
        const modelDir = this.modelPathInput.value.trim();
        if (!modelDir) {
            this.showMessage('Please enter a model directory path', 'warning');
            return;
        }

        this.loadModelBtn.textContent = 'Loading...';
        this.loadModelBtn.disabled = true;

        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_dir: modelDir })
            });

            const data = await response.json();
            this.updateGameDisplay(data);

            if (response.ok) {
                this.modelStatusElement.textContent = '✅ Model loaded successfully';
            } else {
                this.modelStatusElement.textContent = '❌ Model loading failed';
            }
        } catch (error) {
            this.modelStatusElement.textContent = '❌ Connection error';
        } finally {
            this.loadModelBtn.textContent = 'Load Model';
            this.loadModelBtn.disabled = false;
        }
    }

    async resetGame() {
        this.stopAutoPlay();
        
        try {
            const response = await fetch('/api/reset', { method: 'POST' });
            const data = await response.json();
            this.updateGameDisplay(data);
            this.clearHistoryDisplay();
        } catch (error) {
            console.error('Error resetting game:', error);
        }
    }

    setMode(mode) {
        this.currentMode = mode;
        this.stopAutoPlay();

        // Update button states
        this.manualModeBtn.classList.toggle('active', mode === 'manual');
        this.autoModeBtn.classList.toggle('active', mode === 'auto');

        // Show/hide controls
        this.manualControls.style.display = mode === 'manual' ? 'block' : 'none';
        this.autoControls.style.display = mode === 'auto' ? 'flex' : 'none';
    }

    async makeManualMove(direction) {
        if (this.currentMode !== 'manual') return;

        try {
            const response = await fetch('/api/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ direction })
            });

            const data = await response.json();
            this.updateGameDisplay(data);
            this.loadHistory();
        } catch (error) {
            console.error('Error making move:', error);
        }
    }

    async makeAutoMove() {
        try {
            const response = await fetch('/api/auto_move', { method: 'POST' });
            const data = await response.json();
            this.updateGameDisplay(data);

            if (data.done) {
                this.stopAutoPlay();
            }

            this.loadHistory();
            return !data.done;
        } catch (error) {
            console.error('Error making AI move:', error);
            return false;
        }
    }

    async startAutoPlay() {
        if (!this.gameState?.agent_loaded) {
            return; // Silently return if no model loaded
        }

        this.autoPlayBtn.style.display = 'none';
        this.autoStopBtn.style.display = 'inline-block';

        this.autoPlayInterval = setInterval(async () => {
            const canContinue = await this.makeAutoMove();
            if (!canContinue) {
                this.stopAutoPlay();
            }
        }, 500); // 500ms between moves
    }

    stopAutoPlay() {
        if (this.autoPlayInterval) {
            clearInterval(this.autoPlayInterval);
            this.autoPlayInterval = null;
        }

        this.autoPlayBtn.style.display = 'inline-block';
        this.autoStopBtn.style.display = 'none';
    }

    handleKeyboard(event) {
        if (this.currentMode !== 'manual') return;

        const keyMap = {
            'ArrowUp': 'up',
            'ArrowDown': 'down',
            'ArrowLeft': 'left',
            'ArrowRight': 'right',
            'KeyW': 'up',
            'KeyS': 'down',
            'KeyA': 'left',
            'KeyD': 'right'
        };

        const direction = keyMap[event.code];
        if (direction) {
            event.preventDefault();
            this.makeManualMove(direction);
        }
    }

    updateGameDisplay(gameState) {
        this.gameState = gameState;

        // Update stats
        this.scoreElement.textContent = gameState.score?.toLocaleString() || '0';
        this.maxTileElement.textContent = gameState.max_tile || '0';
        this.movesElement.textContent = gameState.moves || '0';
        this.epsilonElement.textContent = gameState.agent_epsilon ? 
            gameState.agent_epsilon.toFixed(4) : '—';

        // Update model status
        if (gameState.agent_loaded) {
            this.modelStatusElement.textContent = '✅ AI model ready';
        } else {
            this.modelStatusElement.textContent = '⚠️ No model loaded';
        }

        // Update board
        this.updateBoard(gameState.board);

        // Handle game over
        if (gameState.done) {
            this.stopAutoPlay();
        }
    }

    updateBoard(board) {
        this.gameBoard.innerHTML = '';

        if (!board || !Array.isArray(board)) {
            // Create empty board
            for (let i = 0; i < 16; i++) {
                const tile = document.createElement('div');
                tile.className = 'tile';
                this.gameBoard.appendChild(tile);
            }
            return;
        }

        board.flat().forEach(value => {
            const tile = document.createElement('div');
            tile.className = 'tile';
            
            if (value > 0) {
                tile.classList.add(`tile-${value}`);
                tile.textContent = value.toLocaleString();
                
                // Add animation for new tiles
                if (this.isNewTile(value)) {
                    tile.classList.add('tile-new');
                }
            }
            
            this.gameBoard.appendChild(tile);
        });
    }

    isNewTile(value) {
        // Simple heuristic: assume tiles with value 2 or 4 might be new
        // This could be improved with more sophisticated tracking
        return value === 2 || value === 4;
    }

    // Messages removed for smooth gameplay

    async loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            this.updateHistoryDisplay(data.history);
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    updateHistoryDisplay(history) {
        if (!history || history.length === 0) {
            this.historyContent.innerHTML = '<div class="history-empty">No moves yet...</div>';
            return;
        }

        const historyHTML = history.slice(-10).reverse().map(move => `
            <div class="history-item">
                <div class="history-move">
                    <span class="history-type ${move.move_type}">${move.move_type}</span>
                    <span>${move.direction.toUpperCase()}</span>
                </div>
                <div class="history-stats">
                    <span>Score: ${move.score}</span>
                    <span>Reward: ${move.reward.toFixed(1)}</span>
                    <span>Max: ${move.max_tile}</span>
                </div>
            </div>
        `).join('');

        this.historyContent.innerHTML = historyHTML;
    }

    clearHistoryDisplay() {
        this.historyContent.innerHTML = '<div class="history-empty">No moves yet...</div>';
    }

    clearHistory() {
        this.clearHistoryDisplay();
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new Game2048DQN();
});