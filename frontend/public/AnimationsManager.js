/**
 * Manages and renders short-lived animations on a canvas context.
 */

import { Constants } from './Constants.js';

export class AnimationsManager {
    constructor() {
        this.animations = new Set();
        this.render = this.render.bind(this);
    }

    /**
     * Register a new animation to be rendered.
     * @param {Function} animationFunction - Function that returns true while animation is active, false when complete
     */
    registerAnimation(animationFunction) {
        if (!animationFunction) return;
        this.animations.add(animationFunction);
    }

    createCollisionAnimation(data) {
        const startTime = performance.now();

        const blinkAnimation = (canvasCtx, timestamp) => {

            canvasCtx.fillStyle = '#cf4c19';
            canvasCtx.beginPath();
            canvasCtx.arc(data.x, data.y, 2, 0, Math.PI * 2);
            canvasCtx.fill();
            // debugger;

            return (timestamp - startTime) / 1000.0 < Constants.COLLISION_ANIMATION_TIME_SEC;
        };

        this.registerAnimation(blinkAnimation);
    }

    /**
     * Creates a blinking animation for a game object.
     * @param {Object} gameObject - The game object to blink
     * @param {number} duration - Duration in milliseconds
     * @param {number} blinkRate - Rate of blinking in milliseconds
     */
    createBlinkAnimation(gameObject, duration, blinkRate = 100) {
        const startTime = performance.now();
        let visible = true;
        let lastToggle = startTime;

        const blinkAnimation = (canvasCtx, timestamp) => {
            if (timestamp - startTime >= duration) {
                return false; // Animation complete
            }

            // Toggle visibility based on blink rate
            if (timestamp - lastToggle >= blinkRate) {
                visible = !visible;
                lastToggle = timestamp;
            }

            if (visible) {
                // TODO player object does not render if it is dead
                gameObject.render(canvasCtx);
            }

            return true; // Animation still active
        };

        this.registerAnimation(blinkAnimation);
    }

    /**
     * Render game objects and all active animations.
     * @param {number} timestamp - Current animation frame timestamp
     */
    render(canvasCtx, timestamp) {

        // Then render all active animations and remove completed ones
        for (const animation of this.animations) {
            const isActive = animation(canvasCtx, timestamp);
            if (!isActive) {
                this.animations.delete(animation);
            }
        }
    }

    reset() {
        this.animations.clear();
    }
}
