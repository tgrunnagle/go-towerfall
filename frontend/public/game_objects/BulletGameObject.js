import { GameObjectWithPosition } from './GameObject.js';
import { Constants } from '../Constants.js';

export class BulletGameObject extends GameObjectWithPosition {

    onCreateAnimation(eventData) {
        const startingX = this.getStatePreferClient('x');
        const dx = this.getStatePreferClient('dx');
        const startingY = this.getStatePreferClient('y');
        const dy = this.getStatePreferClient('dy');
        const startingTime = performance.now();
    
        return (ctx, timestamp) => {
            ctx.beginPath();
            const timeDelta = (timestamp - startingTime) / 1000.0; // Convert to seconds
            const predictedX = startingX + (dx * Constants.BULLET_SPEED_PX_SEC * timeDelta);
            const predictedY = startingY + (dy * Constants.BULLET_SPEED_PX_SEC * timeDelta);
            ctx.arc(predictedX, predictedY, Constants.BULLET_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = '#000000';
            ctx.fill();

            return timeDelta < Constants.BULLET_LIFETIME_SEC;  // Return false when animation is complete
        };
    }

    onDestroyAnimation(eventData) {
        const x = eventData["x"];
        const y = eventData["y"];
        if (x == null || y == null) {
            return null;
        }
        const startingTime = performance.now();

        return (ctx, timestamp) => {
            const timeSeconds = (timestamp - startingTime) / 1000.
            ctx.beginPath();
            const ratio = timeSeconds / Constants.BULLET_DESTROY_ANIMATION_TIME;
            ctx.arc(x, y, Math.max(0.0, ratio * Constants.BULLET_RADIUS), 0, Math.PI * 2);
            ctx.fillStyle = '#ff6666';
            ctx.fill();

            return ratio < 1.0;  // Return false when animation is complete
        };
    }
}

