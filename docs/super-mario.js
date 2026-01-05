class SuperMarioGame {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.mario = null;
        this.gameState = 'ready';
        this.position = 0;
        this.mistakes = 0;
        this.actions = [];
        this.aiActions = [];
        this.criticalFrames = [];
        
        this.gravity = 0.6;
        this.jumpPower = -12;
        this.moveSpeed = 4;
        
        this.cameraX = 0;
        
        this.keys = {
            left: false,
            right: false,
            jump: false
        };
        
        this.platforms = this.generateLevel();
        this.enemies = this.generateEnemies();
        this.coins = this.generateCoins();
        this.flag = { x: 2800, y: 200, height: 150 };
        
        this.clouds = this.generateClouds();
        
        this.setupEventListeners();
        this.render();
    }
    
    generateLevel() {
        const platforms = [];
        
        platforms.push({ x: 0, y: this.height - 40, width: 500, height: 40 });
        platforms.push({ x: 550, y: this.height - 40, width: 300, height: 40 });
        platforms.push({ x: 900, y: this.height - 40, width: 400, height: 40 });
        platforms.push({ x: 1350, y: this.height - 40, width: 600, height: 40 });
        platforms.push({ x: 2000, y: this.height - 40, width: 400, height: 40 });
        platforms.push({ x: 2450, y: this.height - 40, width: 500, height: 40 });
        
        platforms.push({ x: 200, y: this.height - 120, width: 80, height: 20 });
        platforms.push({ x: 350, y: this.height - 160, width: 80, height: 20 });
        platforms.push({ x: 700, y: this.height - 100, width: 100, height: 20 });
        platforms.push({ x: 1000, y: this.height - 140, width: 120, height: 20 });
        platforms.push({ x: 1200, y: this.height - 180, width: 80, height: 20 });
        platforms.push({ x: 1500, y: this.height - 120, width: 100, height: 20 });
        platforms.push({ x: 1700, y: this.height - 160, width: 80, height: 20 });
        platforms.push({ x: 2100, y: this.height - 130, width: 100, height: 20 });
        platforms.push({ x: 2300, y: this.height - 170, width: 80, height: 20 });
        
        platforms.push({ x: 300, y: this.height - 80, width: 40, height: 40 });
        platforms.push({ x: 1100, y: this.height - 80, width: 40, height: 40 });
        platforms.push({ x: 1600, y: this.height - 80, width: 40, height: 40 });
        platforms.push({ x: 2200, y: this.height - 80, width: 40, height: 40 });
        
        return platforms;
    }
    
    generateEnemies() {
        return [
            { x: 400, y: this.height - 70, width: 30, height: 30, vx: -1, type: 'goomba' },
            { x: 800, y: this.height - 70, width: 30, height: 30, vx: -1, type: 'goomba' },
            { x: 1150, y: this.height - 70, width: 30, height: 30, vx: -1, type: 'goomba' },
            { x: 1450, y: this.height - 70, width: 30, height: 30, vx: -1, type: 'goomba' },
            { x: 1850, y: this.height - 70, width: 30, height: 30, vx: -1, type: 'goomba' },
            { x: 2250, y: this.height - 70, width: 30, height: 30, vx: -1, type: 'goomba' },
        ];
    }
    
    generateCoins() {
        return [
            { x: 220, y: this.height - 150, collected: false },
            { x: 370, y: this.height - 190, collected: false },
            { x: 720, y: this.height - 130, collected: false },
            { x: 1020, y: this.height - 170, collected: false },
            { x: 1520, y: this.height - 150, collected: false },
            { x: 1720, y: this.height - 190, collected: false },
            { x: 2120, y: this.height - 160, collected: false },
        ];
    }
    
    generateClouds() {
        const clouds = [];
        for (let i = 0; i < 15; i++) {
            clouds.push({
                x: Math.random() * 3000,
                y: 30 + Math.random() * 80,
                width: 60 + Math.random() * 40,
                speed: 0.2 + Math.random() * 0.3
            });
        }
        return clouds;
    }
    
    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            if (this.gameState !== 'playing' && this.gameState !== 'practice') return;
            
            if (e.key === 'ArrowRight' || e.key === 'd') this.keys.right = true;
            if (e.key === 'ArrowLeft' || e.key === 'a') this.keys.left = true;
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === ' ') {
                this.keys.jump = true;
                e.preventDefault();
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.key === 'ArrowRight' || e.key === 'd') this.keys.right = false;
            if (e.key === 'ArrowLeft' || e.key === 'a') this.keys.left = false;
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === ' ') this.keys.jump = false;
        });
        
        document.getElementById('sm-start').addEventListener('click', () => this.start());
        document.getElementById('sm-tutorial').addEventListener('click', () => this.startTutorial());
        document.getElementById('sm-practice').addEventListener('click', () => this.startPractice());
    }
    
    start() {
        this.reset();
        this.gameState = 'playing';
        this.actions = [];
        this.aiActions = [];
        this.criticalFrames = [];
        document.getElementById('sm-status').textContent = 'Playing';
        this.gameLoop();
    }
    
    reset() {
        this.mario = {
            x: 50,
            y: this.height - 100,
            vx: 0,
            vy: 0,
            width: 25,
            height: 35,
            onGround: false,
            facingRight: true
        };
        this.cameraX = 0;
        this.position = 0;
        this.mistakes = 0;
        
        this.enemies = this.generateEnemies();
        this.coins = this.generateCoins();
        
        document.getElementById('sm-position').textContent = '0';
        document.getElementById('sm-mistakes').textContent = '0';
    }
    
    getOptimalAction(mario, enemies) {
        let action = 1;
        
        const nearestEnemy = enemies.find(e => {
            const dist = e.x - mario.x;
            return dist > 0 && dist < 150;
        });
        
        if (nearestEnemy) {
            const dist = nearestEnemy.x - mario.x;
            if (dist < 80 && dist > 30 && mario.onGround) {
                return 4;
            }
        }
        
        const pit = this.checkForPit(mario.x + 50);
        if (pit && mario.onGround) {
            return 4;
        }
        
        return action;
    }
    
    checkForPit(x) {
        for (const platform of this.platforms) {
            if (x >= platform.x && x <= platform.x + platform.width && platform.y >= this.height - 50) {
                return false;
            }
        }
        return true;
    }
    
    getCurrentAction() {
        if (this.keys.right && this.keys.jump) return 4;
        if (this.keys.left && this.keys.jump) return 3;
        if (this.keys.jump) return 5;
        if (this.keys.right) return 1;
        if (this.keys.left) return 6;
        return 0;
    }
    
    update() {
        if (this.gameState !== 'playing' && this.gameState !== 'tutorial' && this.gameState !== 'practice') return;
        
        const mario = this.mario;
        let action;
        
        if (this.gameState === 'tutorial' && this.tutorialFrame !== undefined) {
            if (this.tutorialFrame < this.tutorialTakeoverFrame) {
                action = this.actions[this.tutorialFrame] || 0;
            } else {
                action = this.getOptimalAction(mario, this.enemies);
            }
            this.tutorialFrame++;
        } else if (this.gameState === 'practice' && this.practiceFrame !== undefined) {
            if (this.practiceFrame < this.practiceTakeoverFrame) {
                action = this.actions[this.practiceFrame] || 0;
            } else {
                action = this.getCurrentAction();
            }
            this.practiceFrame++;
        } else {
            action = this.getCurrentAction();
            this.actions.push(action);
            
            const optimalAction = this.getOptimalAction(mario, this.enemies);
            this.aiActions.push(optimalAction);
            
            const nearEnemy = this.enemies.some(e => Math.abs(e.x - mario.x) < 100);
            const nearPit = this.checkForPit(mario.x + 50);
            
            if ((nearEnemy || nearPit) && action !== optimalAction) {
                this.criticalFrames.push(this.actions.length - 1);
                this.mistakes++;
                document.getElementById('sm-mistakes').textContent = this.mistakes;
            }
        }
        
        mario.vx = 0;
        if (action === 1 || action === 4) {
            mario.vx = this.moveSpeed;
            mario.facingRight = true;
        }
        if (action === 6 || action === 3) {
            mario.vx = -this.moveSpeed;
            mario.facingRight = false;
        }
        if ((action === 5 || action === 4 || action === 3) && mario.onGround) {
            mario.vy = this.jumpPower;
            mario.onGround = false;
        }
        
        mario.vy += this.gravity;
        mario.x += mario.vx;
        mario.y += mario.vy;
        
        mario.onGround = false;
        for (const platform of this.platforms) {
            if (this.checkCollision(mario, platform)) {
                if (mario.vy > 0) {
                    mario.y = platform.y - mario.height;
                    mario.vy = 0;
                    mario.onGround = true;
                }
            }
        }
        
        if (mario.y > this.height) {
            document.getElementById('sm-status').textContent = 'Fell! 💀';
            this.endGame(false);
            return;
        }
        
        for (const enemy of this.enemies) {
            enemy.x += enemy.vx;
            
            if (this.checkCollision(mario, enemy)) {
                if (mario.vy > 0 && mario.y + mario.height - 10 < enemy.y) {
                    this.enemies = this.enemies.filter(e => e !== enemy);
                    mario.vy = -8;
                } else {
                    document.getElementById('sm-status').textContent = 'Hit! 💀';
                    this.endGame(false);
                    return;
                }
            }
        }
        
        for (const coin of this.coins) {
            if (!coin.collected && this.checkCollision(mario, { x: coin.x - 10, y: coin.y - 10, width: 20, height: 20 })) {
                coin.collected = true;
            }
        }
        
        this.position = Math.round(mario.x);
        document.getElementById('sm-position').textContent = this.position;
        
        this.cameraX = Math.max(0, mario.x - this.width / 3);
        
        if (mario.x >= this.flag.x) {
            document.getElementById('sm-status').textContent = 'Won! 🏆';
            this.endGame(true);
        }
        
        this.clouds.forEach(cloud => {
            cloud.x -= cloud.speed;
            if (cloud.x + cloud.width < 0) {
                cloud.x = 3000;
            }
        });
    }
    
    checkCollision(a, b) {
        return a.x < b.x + b.width &&
               a.x + a.width > b.x &&
               a.y < b.y + b.height &&
               a.y + a.height > b.y;
    }
    
    endGame(success) {
        this.gameState = 'ended';
        
        if (this.criticalFrames.length > 0) {
            document.getElementById('sm-tutorial').disabled = false;
            document.getElementById('sm-practice').disabled = false;
        }
    }
    
    startTutorial() {
        if (this.criticalFrames.length === 0) return;
        
        this.reset();
        this.gameState = 'tutorial';
        this.tutorialFrame = 0;
        this.tutorialTakeoverFrame = Math.max(0, this.criticalFrames[this.criticalFrames.length - 1] - 20);
        document.getElementById('sm-status').textContent = 'Tutorial Mode';
        this.gameLoop();
    }
    
    startPractice() {
        if (this.criticalFrames.length === 0) return;
        
        this.reset();
        this.gameState = 'practice';
        this.practiceFrame = 0;
        this.practiceTakeoverFrame = Math.max(0, this.criticalFrames[this.criticalFrames.length - 1] - 20);
        document.getElementById('sm-status').textContent = 'Practice Mode';
        this.gameLoop();
    }
    
    render() {
        const ctx = this.ctx;
        
        const gradient = ctx.createLinearGradient(0, 0, 0, this.height);
        gradient.addColorStop(0, '#5c94fc');
        gradient.addColorStop(1, '#87ceeb');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, this.width, this.height);
        
        this.clouds.forEach(cloud => {
            const screenX = cloud.x - this.cameraX * 0.3;
            if (screenX > -100 && screenX < this.width + 100) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.beginPath();
                ctx.arc(screenX, cloud.y, cloud.width / 3, 0, Math.PI * 2);
                ctx.arc(screenX + cloud.width / 3, cloud.y - 10, cloud.width / 4, 0, Math.PI * 2);
                ctx.arc(screenX + cloud.width / 2, cloud.y, cloud.width / 3, 0, Math.PI * 2);
                ctx.fill();
            }
        });
        
        ctx.save();
        ctx.translate(-this.cameraX, 0);
        
        for (const platform of this.platforms) {
            if (platform.x + platform.width > this.cameraX && platform.x < this.cameraX + this.width) {
                if (platform.height > 30) {
                    ctx.fillStyle = '#8B4513';
                    ctx.fillRect(platform.x, platform.y, platform.width, platform.height);
                    ctx.fillStyle = '#228B22';
                    ctx.fillRect(platform.x, platform.y, platform.width, 10);
                } else if (platform.width === 40) {
                    ctx.fillStyle = '#DAA520';
                    ctx.fillRect(platform.x, platform.y, platform.width, platform.height);
                    ctx.fillStyle = '#B8860B';
                    ctx.fillRect(platform.x + 5, platform.y + 5, 10, 10);
                    ctx.fillRect(platform.x + 25, platform.y + 25, 10, 10);
                } else {
                    ctx.fillStyle = '#8B4513';
                    ctx.fillRect(platform.x, platform.y, platform.width, platform.height);
                }
            }
        }
        
        for (const coin of this.coins) {
            if (!coin.collected) {
                ctx.fillStyle = '#FFD700';
                ctx.beginPath();
                ctx.arc(coin.x, coin.y, 10, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#FFA500';
                ctx.beginPath();
                ctx.arc(coin.x, coin.y, 6, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        
        for (const enemy of this.enemies) {
            ctx.fillStyle = '#8B4513';
            ctx.beginPath();
            ctx.arc(enemy.x + enemy.width / 2, enemy.y + enemy.height / 2, enemy.width / 2, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(enemy.x + 8, enemy.y + 8, 4, 0, Math.PI * 2);
            ctx.arc(enemy.x + 22, enemy.y + 8, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#000';
            ctx.beginPath();
            ctx.arc(enemy.x + 8, enemy.y + 8, 2, 0, Math.PI * 2);
            ctx.arc(enemy.x + 22, enemy.y + 8, 2, 0, Math.PI * 2);
            ctx.fill();
        }
        
        const flagX = this.flag.x;
        ctx.fillStyle = '#654321';
        ctx.fillRect(flagX, this.flag.y, 8, this.flag.height);
        ctx.fillStyle = '#00FF00';
        ctx.beginPath();
        ctx.moveTo(flagX + 8, this.flag.y);
        ctx.lineTo(flagX + 50, this.flag.y + 25);
        ctx.lineTo(flagX + 8, this.flag.y + 50);
        ctx.closePath();
        ctx.fill();
        
        if (this.mario) {
            const mario = this.mario;
            
            ctx.fillStyle = '#FF0000';
            ctx.fillRect(mario.x + 5, mario.y, 15, 10);
            
            ctx.fillStyle = '#FFD8B1';
            ctx.fillRect(mario.x + 5, mario.y + 10, 15, 10);
            
            ctx.fillStyle = '#0000FF';
            ctx.fillRect(mario.x + 3, mario.y + 20, 19, 15);
            
            ctx.fillStyle = '#8B4513';
            ctx.fillRect(mario.x + 3, mario.y + 30, 8, 5);
            ctx.fillRect(mario.x + 14, mario.y + 30, 8, 5);
            
            ctx.fillStyle = '#000';
            const eyeX = mario.facingRight ? mario.x + 15 : mario.x + 8;
            ctx.fillRect(eyeX, mario.y + 12, 3, 3);
        }
        
        ctx.restore();
        
        if (this.gameState === 'tutorial' && this.tutorialFrame >= this.tutorialTakeoverFrame) {
            ctx.fillStyle = 'rgba(0, 212, 255, 0.3)';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🤖 AI DEMONSTRATING', this.width / 2, 40);
            ctx.textAlign = 'left';
        }
        
        if (this.gameState === 'practice' && this.practiceFrame >= this.practiceTakeoverFrame) {
            ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🎮 YOUR TURN!', this.width / 2, 40);
            ctx.textAlign = 'left';
        }
    }
    
    gameLoop() {
        if (this.gameState === 'playing' || this.gameState === 'tutorial' || this.gameState === 'practice') {
            this.update();
            this.render();
            requestAnimationFrame(() => this.gameLoop());
        } else {
            this.render();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new SuperMarioGame('mario-canvas');
});
