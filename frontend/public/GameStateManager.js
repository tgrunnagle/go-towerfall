import { PlayerGameObject } from './game_objects/PlayerGameObject.js';
import { BulletGameObject } from './game_objects/BulletGameObject.js';
import { BlockGameObject } from './game_objects/BlockGameObject.js';
import { ArrowGameObject } from './game_objects/ArrowGameObject.js';
import { AnimationsManager } from './AnimationsManager.js';
import { Constants } from './Constants.js';

export class GameStateManager {
    constructor() {
        this.gameObjects = {};
        this.spectators = [];
        this.currentPlayerObjectId = null;

        this.setCurrentPlayerObjectId = this.setCurrentPlayerObjectId.bind(this);
        this.handleGameStateUpdate = this.handleGameStateUpdate.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.getCurrentPlayerClientState = this.getCurrentPlayerClientState.bind(this);
        this.reset = this.reset.bind(this);
        this.drawSpectators = this.drawSpectators.bind(this);
        this.animationManager = new AnimationsManager();

        this.canvasSizeX = 0;
        this.canvasSizeY = 0;
    }

    setCanvasSize(canvasSizeX, canvasSizeY) {
        this.canvasSizeX = canvasSizeX;
        this.canvasSizeY = canvasSizeY;
    }

    reset() {
        this.gameObjects = {};
        this.currentPlayerObjectId = null;
        this.animationManager.reset();
    }

    setCurrentPlayerObjectId(id) {
        this.currentPlayerObjectId = id;
        if (this.currentPlayerObjectId in this.gameObjects) {
            this.gameObjects[this.currentPlayerObjectId].setClientState(true, 'isCurrentPlayer');
        }
    }

    handleGameStateUpdate(payload) {
        if (payload.objectStates) {
            // Remove items not in the payload if this is a full update
            if (payload.fullUpdate) {
                Object.keys(this.gameObjects).forEach(function (objectId) {
                    if (!(objectId in payload.objectStates)) {
                        delete this.gameObjects[objectId];
                    }
                }, this);
            }

            // Process the payload
            Object.keys(payload.objectStates).forEach(function (objectId) {
                // Create new objects
                if (!(objectId in this.gameObjects)) {
                    // null values indicate an object was destroyed, so we can skip it here
                    if (!payload.objectStates[objectId]) return;

                    switch (payload.objectStates[objectId].objectType) {
                        case 'player':
                            this.gameObjects[objectId] = new PlayerGameObject(
                                {
                                    ...payload.objectStates[objectId],
                                },
                            );
                            break;
                        case 'bullet':
                            this.gameObjects[objectId] = new BulletGameObject(
                                {
                                    ...payload.objectStates[objectId],
                                },
                            );
                            break;
                        case 'block':
                            this.gameObjects[objectId] = new BlockGameObject(
                                {
                                    ...payload.objectStates[objectId],
                                },
                            );
                            break;
                        case 'arrow':
                            this.gameObjects[objectId] = new ArrowGameObject(
                                {
                                    ...payload.objectStates[objectId],
                                },
                            );
                            break;
                        default:
                            console.error('Unknown object type: ' + payload.objectStates[objectId].objectType);
                            return;
                    }
                }
                // Update existing objects
                else {
                    // update only the server state
                    this.gameObjects[objectId].setServerState({
                        ...payload.objectStates[objectId],
                    });
                }

                // Check if this object is the current player
                if (objectId === this.currentPlayerObjectId) {
                    this.gameObjects[objectId].setClientState(true, 'isCurrentPlayer');
                }
            }, this);
        } else {
            // For full updates with no game objects, reset all objects
            if (payload.fullUpdate) {
                delete this.gameObjects;
                this.gameObjects = {};
            }
            this.animationManager.reset();
        }

        if (payload.events) {
            payload.events.forEach(event => {
                switch (event.type) {
                    case "object_created":
                        const createdId = event.data["objectID"];
                        this.animationManager.registerAnimation(this.gameObjects[createdId].onCreateAnimation?.(event.data));
                        break;
                    case "object_destroyed":
                        const destroyedId = event.data["objectID"];
                        this.animationManager.registerAnimation(this.gameObjects[destroyedId].onDestroyAnimation?.(event.data));
                        delete this.gameObjects[destroyedId];
                        break;
                    case "player_died":
                        const deadPlayerId = event.data["objectID"];
                        this.animationManager.registerAnimation(this.gameObjects[deadPlayerId].onDiedAnimation?.(event.data));
                        break;
                    case "collision":
                        this.animationManager.registerAnimation(this.animationManager.createCollisionAnimation(event.data));
                        break;
                    default:
                        console.error('Unknown event type: ' + event.type);
                        return;
                }
            });
        }
    }

    handleSpectatorsUpdate(payload) {
        if (payload.spectators) {
            this.spectators = payload.spectators;
        }
    }

    render(canvasCtx, timestamp) {
        // Clear canvas
        canvasCtx.clearRect(0, 0, this.canvasSizeX, this.canvasSizeY);

        // Draw grid lines
        canvasCtx.strokeStyle = '#e0e0e0';
        canvasCtx.lineWidth = 1;

        // Draw vertical grid lines
        for (let x = 0; x <= this.canvasSizeX; x += 64) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(x, 0);
            canvasCtx.lineTo(x, this.canvasSizeY);
            canvasCtx.stroke();
        }

        // Draw horizontal grid lines
        for (let y = 0; y <= this.canvasSizeY; y += 64) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(0, y);
            canvasCtx.lineTo(this.canvasSizeX, y);
            canvasCtx.stroke();
        }

        Object.values(this.gameObjects).forEach(gameObject => {
            gameObject.render(canvasCtx, timestamp);
        });
        this.animationManager.render(canvasCtx, timestamp);

        this.drawSpectators(canvasCtx);
    }

    drawSpectators(canvasCtx) {
        if (!this.spectators || this.spectators.length === 0) return;
        canvasCtx.fillStyle = Constants.SPECTATOR_TEXT_COLOR;
        canvasCtx.font = Constants.SPECTATOR_TEXT_FONT;
        canvasCtx.textBaseline = 'top';
        canvasCtx.textAlign = 'left';
        canvasCtx.fillText('Spectators:', Constants.SPECTATOR_TEXT_OFFSET_X, Constants.SPECTATOR_TEXT_OFFSET_Y);
        this.spectators.forEach((spectator, index) => {
            canvasCtx.fillText(
                spectator,
                Constants.SPECTATOR_TEXT_OFFSET_X,
                Constants.SPECTATOR_TEXT_OFFSET_Y + ((index + 1) * Constants.SPECTATOR_TEXT_LINE_HEIGHT)
            );
        });
    }

    handleMouseMove(x, y) {
        if (!this.currentPlayerObjectId || !(this.currentPlayerObjectId in this.gameObjects)) return;

        const dX = x - this.gameObjects[this.currentPlayerObjectId].getStatePreferClient('x');
        const dY = y - this.gameObjects[this.currentPlayerObjectId].getStatePreferClient('y');
        this.gameObjects[this.currentPlayerObjectId].setClientState(Math.atan2(dY, dX), 'dir');
    }

    getCurrentPlayerClientState() {
        return this.currentPlayerObjectId && this.currentPlayerObjectId in this.gameObjects
            ? { ...this.gameObjects[this.currentPlayerObjectId].clientState }
            : null;
    }
}

export default GameStateManager;