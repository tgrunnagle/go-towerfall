import { GameObjectWithPosition } from "./GameObject.js";

export class BlockGameObject extends GameObjectWithPosition {
    constructor(serverState, clientState) {
        super(serverState, clientState);
        this.render = this.render.bind(this);
    }

    render(canvasCtx, _timestamp) {
        if (!this.serverState.pts || this.serverState.pts.length < 2) {
            return;
        }
        canvasCtx.fillStyle = '#009900';
        canvasCtx.beginPath();
        canvasCtx.moveTo(this.serverState.pts[0].X, this.serverState.pts[0].Y);
        for (let i = 1; i < this.serverState.pts.length; i++) {
            canvasCtx.lineTo(this.serverState.pts[i].X, this.serverState.pts[i].Y);
        }
        canvasCtx.closePath();
        canvasCtx.fill();
    }
}