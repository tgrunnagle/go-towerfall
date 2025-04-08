import { GameObjectWithPosition } from "./GameObject.js";

export class BlockGameObject extends GameObjectWithPosition {
    constructor(serverState, clientState) {
        super(serverState, clientState);
        this.render = this.render.bind(this);
    }

    render(canvasCtx, _timestamp) {
        canvasCtx.beginPath();
        canvasCtx.rect(this.serverState.x, this.serverState.y, this.serverState.w, this.serverState.h);
        canvasCtx.fillStyle = '#009900';
        canvasCtx.fill();
    }
}