
export const defaultClientState = () => {
    return {
        isCurrentPlayer: false,
        dir: Math.PI * 3 / 2,
        lastUpdateFromServer: performance.now(),
    };
}

export class GameObject {
    constructor(serverState, clientState) {
        this.serverState = serverState ? { ...serverState } : {};
        this.clientState = clientState ? { ...clientState } : defaultClientState();
        this.getStatePreferClient = this.getStatePreferClient.bind(this);
        this.setServerState = this.setServerState.bind(this);
        this.setClientState = this.setClientState.bind(this);
        this.render = this.render.bind(this);
    }

    getStatePreferClient(key) {
        return this.clientState[key] == null ? this.serverState[key] : this.clientState[key];
    }

    setServerState(state, key = null) {
        if (key) {
            this.serverState[key] = state;
            return;
        }
        this.serverState = state
            ? {
                ...state,
            }
            : {};
        if (key) return;
        // only update rendersSinceUpdate if is a full state update
        this.setClientState(0, 'rendersSinceUpdate');
        this.setClientState(performance.now(), 'lastUpdateFromServer');
    }

    setClientState(state, key = null) {
        if (key) {
            this.clientState[key] = state;
            return;
        }
        this.clientState = state
            ? { ...state }
            : {};
    }

    render() {
        return;
    }

    onCreateAnimation(eventData) {
        return null;
    }

    onDestroyAnimation(eventData) {
        return null;
    }
}

export class GameObjectWithPosition extends GameObject {
    constructor(serverState, clientState, positionInterpolationSpeed = 0.2, directionInterpolationSpeed = 0.1) {
        super(serverState, clientState);
        this.positionInterpolationSpeed = positionInterpolationSpeed;
        this.directionInterpolationSpeed = directionInterpolationSpeed;
        if (!this.clientState.x) {
            this.clientState.x = this.serverState.x;
        }
        if (!this.clientState.y) {
            this.clientState.y = this.serverState.y;
        }
        if (!this.clientState.dir) {
            this.clientState.dir = this.serverState.dir;
        }

        this.predictPosition = this.interpPosition.bind(this);
        this.predictDirection = this.interpDirection.bind(this);
    }
        
    interpPosition(timestamp) {
        if (this.clientState.x == null) {
            this.clientState.x = this.serverState.x;
        }
        if (this.clientState.y == null) {
            this.clientState.y = this.serverState.y;
        }

        // Update interpolated position
        const shouldInterp = this.clientState.rendersSinceUpdate >= (1. / this.positionInterpolationSpeed);
        this.clientState.x += (this.serverState.x - this.clientState.x) * (shouldInterp ? this.positionInterpolationSpeed : 1);
        this.clientState.y += (this.serverState.y - this.clientState.y) * (shouldInterp ? this.positionInterpolationSpeed : 1);

        // Apply velocity for prediction
        const timeDelta = (timestamp - this.clientState.lastUpdateFromServer) / 1000; // Convert to seconds
        const predictedX = this.clientState.x + (this.serverState.dx || 0) * timeDelta;
        const predictedY = this.clientState.y + (this.serverState.dy || 0) * timeDelta;

        return { predictedX, predictedY };
    }

    interpDirection() {
        if (this.clientState.dir == null) {
            this.clientState.dir = this.serverState.dir;
        }

        if (this.clientState.dir == null) {
            return null;
        }

        if (this.clientState.isCurrentPlayer) {
            return this.clientState.dir;
        }

        const diff = this.serverState.dir - this.clientState.dir;
        if (diff === 0.0) {
            return this.clientState.dir;
        }

        // Update interpolated direction
        const shouldInterp = this.clientState.rendersSinceUpdate >= (1. / this.directionInterpolationSpeed);
        this.clientState.dir += diff * (shouldInterp ? this.directionInterpolationSpeed : 1);
        return this.clientState.dir;
    }
}
