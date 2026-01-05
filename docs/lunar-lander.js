class LunarLanderGame {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.lander = null;
        this.gameState = 'ready';
        this.score = 0;
        this.mistakes = 0;
        this.actions = [];
        this.aiActions = [];
        this.criticalFrames = [];
        
        this.gravity = 0.05;
        this.thrustPower = 0.12;
        this.sideThrustPower = 0.08;
        
        this.keys = {
            up: false,
            left: false,
            right: false
        };
        
        this.landingPad = {
            x: this.width / 2 - 50,
            y: this.height - 30,
            width: 100,
            height: 10
        };
        
        this.terrain = this.generateTerrain();
        this.stars = this.generateStars();
        
        this.setupEventListeners();
        this.render();
    }
    
    generateTerrain() {
        const points = [];
        const segments = 20;
        const segmentWidth = this.width / segments;
        
        for (let i = 0; i <= segments; i++) {
            const x = i * segmentWidth;
            let y;
            
            if (i >= 8 && i <= 12) {
                y = this.height - 30;
            } else {
                y = this.height - 30 - Math.random() * 60 - 20;
            }
            
            points.push({ x, y });
        }
        
        return points;
    }
    
    generateStars() {
        const stars = [];
        for (let i = 0; i < 100; i++) {
            stars.push({
                x: Math.random() * this.width,
                y: Math.random() * (this.height - 100),
                size: Math.random() * 2 + 0.5,
                brightness: Math.random()
            });
        }
        return stars;
    }
    
    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            if (this.gameState !== 'playing') return;
            
            if (e.key === 'ArrowUp' || e.key === 'w') this.keys.up = true;
            if (e.key === 'ArrowLeft' || e.key === 'a') this.keys.left = true;
            if (e.key === 'ArrowRight' || e.key === 'd') this.keys.right = true;
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.key === 'ArrowUp' || e.key === 'w') this.keys.up = false;
            if (e.key === 'ArrowLeft' || e.key === 'a') this.keys.left = false;
            if (e.key === 'ArrowRight' || e.key === 'd') this.keys.right = false;
        });
        
        document.getElementById('ll-start').addEventListener('click', () => this.start());
        document.getElementById('ll-tutorial').addEventListener('click', () => this.startTutorial());
        document.getElementById('ll-practice').addEventListener('click', () => this.startPractice());
    }
    
    start() {
        this.reset();
        this.gameState = 'playing';
        this.actions = [];
        this.aiActions = [];
        this.criticalFrames = [];
        document.getElementById('ll-status').textContent = 'Playing';
        this.gameLoop();
    }
    
    reset() {
        this.lander = {
            x: this.width / 2,
            y: 60,
            vx: (Math.random() - 0.5) * 2,
            vy: 0,
            angle: 0,
            fuel: 100,
            width: 30,
            height: 40
        };
        this.score = 0;
        this.mistakes = 0;
        document.getElementById('ll-score').textContent = '0';
        document.getElementById('ll-mistakes').textContent = '0';
    }
    
    getOptimalAction(lander) {
        const targetX = this.landingPad.x + this.landingPad.width / 2;
        const distX = targetX - lander.x;
        const distY = this.landingPad.y - lander.y;
        
        let action = 0;
        
        if (lander.vy > 1.5 || (distY < 100 && lander.vy > 0.8)) {
            action = 2;
        }
        
        if (Math.abs(distX) > 30) {
            if (distX > 0 && lander.vx < 1) {
                action = 3;
            } else if (distX < 0 && lander.vx > -1) {
                action = 1;
            }
        } else {
            if (lander.vx > 0.3) {
                action = 1;
            } else if (lander.vx < -0.3) {
                action = 3;
            }
        }
        
        if (lander.vy > 2) {
            action = 2;
        }
        
        return action;
    }
    
    getCurrentAction() {
        if (this.keys.up) return 2;
        if (this.keys.left) return 1;
        if (this.keys.right) return 3;
        return 0;
    }
    
    update() {
        if (this.gameState !== 'playing' && this.gameState !== 'tutorial' && this.gameState !== 'practice') return;
        
        const lander = this.lander;
        let action;
        
        if (this.gameState === 'tutorial' && this.tutorialFrame !== undefined) {
            if (this.tutorialFrame < this.tutorialTakeoverFrame) {
                action = this.actions[this.tutorialFrame] || 0;
            } else {
                action = this.getOptimalAction(lander);
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
            
            const optimalAction = this.getOptimalAction(lander);
            this.aiActions.push(optimalAction);
            
            const isImportant = Math.abs(lander.vy) > 1 || Math.abs(lander.vx) > 0.5 || 
                               (this.landingPad.y - lander.y < 150);
            
            if (isImportant && action !== optimalAction) {
                this.criticalFrames.push(this.actions.length - 1);
                this.mistakes++;
                document.getElementById('ll-mistakes').textContent = this.mistakes;
            }
        }
        
        lander.vy += this.gravity;
        
        if (action === 2 && lander.fuel > 0) {
            lander.vy -= this.thrustPower;
            lander.fuel -= 0.5;
        }
        if (action === 1 && lander.fuel > 0) {
            lander.vx -= this.sideThrustPower;
            lander.fuel -= 0.2;
        }
        if (action === 3 && lander.fuel > 0) {
            lander.vx += this.sideThrustPower;
            lander.fuel -= 0.2;
        }
        
        lander.x += lander.vx;
        lander.y += lander.vy;
        
        if (lander.x < 0) { lander.x = 0; lander.vx = 0; }
        if (lander.x > this.width) { lander.x = this.width; lander.vx = 0; }
        
        this.checkCollision();
    }
    
    checkCollision() {
        const lander = this.lander;
        const pad = this.landingPad;
        
        if (lander.y + lander.height / 2 >= pad.y) {
            const onPad = lander.x >= pad.x && lander.x <= pad.x + pad.width;
            const softLanding = Math.abs(lander.vy) < 1.5 && Math.abs(lander.vx) < 0.5;
            
            if (onPad && softLanding) {
                this.score = Math.round(100 + lander.fuel * 2 - this.mistakes * 10);
                document.getElementById('ll-score').textContent = Math.max(0, this.score);
                document.getElementById('ll-status').textContent = 'Landed! 🎉';
                this.endGame(true);
            } else {
                this.score = -50;
                document.getElementById('ll-score').textContent = this.score;
                document.getElementById('ll-status').textContent = 'Crashed! 💥';
                this.endGame(false);
            }
        }
        
        for (let i = 0; i < this.terrain.length - 1; i++) {
            const p1 = this.terrain[i];
            const p2 = this.terrain[i + 1];
            
            if (lander.x >= p1.x && lander.x <= p2.x) {
                const terrainY = p1.y + (p2.y - p1.y) * (lander.x - p1.x) / (p2.x - p1.x);
                if (lander.y + lander.height / 2 >= terrainY) {
                    this.score = -50;
                    document.getElementById('ll-score').textContent = this.score;
                    document.getElementById('ll-status').textContent = 'Crashed! 💥';
                    this.endGame(false);
                }
            }
        }
    }
    
    endGame(success) {
        this.gameState = 'ended';
        
        if (this.criticalFrames.length > 0) {
            document.getElementById('ll-tutorial').disabled = false;
            document.getElementById('ll-practice').disabled = false;
        }
    }
    
    startTutorial() {
        if (this.criticalFrames.length === 0) return;
        
        this.reset();
        this.gameState = 'tutorial';
        this.tutorialFrame = 0;
        this.tutorialTakeoverFrame = Math.max(0, this.criticalFrames[this.criticalFrames.length - 1] - 30);
        document.getElementById('ll-status').textContent = 'Tutorial Mode';
        this.gameLoop();
    }
    
    startPractice() {
        if (this.criticalFrames.length === 0) return;
        
        this.reset();
        this.gameState = 'practice';
        this.practiceFrame = 0;
        this.practiceTakeoverFrame = Math.max(0, this.criticalFrames[this.criticalFrames.length - 1] - 30);
        document.getElementById('ll-status').textContent = 'Practice Mode';
        this.gameLoop();
    }
    
    render() {
        const ctx = this.ctx;
        
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, this.width, this.height);
        
        this.stars.forEach(star => {
            const flicker = 0.7 + Math.sin(Date.now() / 500 + star.brightness * 10) * 0.3;
            ctx.fillStyle = `rgba(255, 255, 255, ${star.brightness * flicker})`;
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
            ctx.fill();
        });
        
        ctx.fillStyle = '#2a2a3a';
        ctx.beginPath();
        ctx.moveTo(this.terrain[0].x, this.height);
        this.terrain.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.lineTo(this.width, this.height);
        ctx.closePath();
        ctx.fill();
        
        ctx.strokeStyle = '#4a4a5a';
        ctx.lineWidth = 2;
        ctx.beginPath();
        this.terrain.forEach((p, i) => {
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
        
        const pad = this.landingPad;
        ctx.fillStyle = '#00ff00';
        ctx.fillRect(pad.x, pad.y, pad.width, pad.height);
        
        ctx.fillStyle = '#ffff00';
        ctx.fillRect(pad.x, pad.y - 5, 5, 15);
        ctx.fillRect(pad.x + pad.width - 5, pad.y - 5, 5, 15);
        
        if (this.lander) {
            const lander = this.lander;
            ctx.save();
            ctx.translate(lander.x, lander.y);
            
            ctx.fillStyle = '#c0c0c0';
            ctx.beginPath();
            ctx.moveTo(0, -lander.height / 2);
            ctx.lineTo(-lander.width / 2, lander.height / 2);
            ctx.lineTo(lander.width / 2, lander.height / 2);
            ctx.closePath();
            ctx.fill();
            
            ctx.fillStyle = '#00d4ff';
            ctx.beginPath();
            ctx.arc(0, -lander.height / 4, 8, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.strokeStyle = '#808080';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(-lander.width / 2, lander.height / 2);
            ctx.lineTo(-lander.width / 2 - 5, lander.height / 2 + 10);
            ctx.moveTo(lander.width / 2, lander.height / 2);
            ctx.lineTo(lander.width / 2 + 5, lander.height / 2 + 10);
            ctx.stroke();
            
            if (this.keys.up || (this.gameState === 'tutorial' && this.getOptimalAction(lander) === 2)) {
                ctx.fillStyle = `rgba(255, ${100 + Math.random() * 100}, 0, ${0.7 + Math.random() * 0.3})`;
                ctx.beginPath();
                ctx.moveTo(-10, lander.height / 2);
                ctx.lineTo(10, lander.height / 2);
                ctx.lineTo(0, lander.height / 2 + 20 + Math.random() * 15);
                ctx.closePath();
                ctx.fill();
            }
            
            ctx.restore();
            
            ctx.fillStyle = '#00d4ff';
            ctx.fillRect(10, 10, lander.fuel * 1.5, 15);
            ctx.strokeStyle = '#00d4ff';
            ctx.strokeRect(10, 10, 150, 15);
            ctx.fillStyle = '#fff';
            ctx.font = '12px Arial';
            ctx.fillText('FUEL', 170, 22);
            
            ctx.fillStyle = '#fff';
            ctx.font = '14px Arial';
            ctx.fillText(`VX: ${lander.vx.toFixed(2)}`, 10, 45);
            ctx.fillText(`VY: ${lander.vy.toFixed(2)}`, 10, 60);
        }
        
        if (this.gameState === 'tutorial' && this.tutorialFrame >= this.tutorialTakeoverFrame) {
            ctx.fillStyle = 'rgba(0, 212, 255, 0.3)';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🤖 AI DEMONSTRATING', this.width / 2, 80);
            ctx.textAlign = 'left';
        }
        
        if (this.gameState === 'practice' && this.practiceFrame >= this.practiceTakeoverFrame) {
            ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🎮 YOUR TURN!', this.width / 2, 80);
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
    new LunarLanderGame('lunar-canvas');
});
