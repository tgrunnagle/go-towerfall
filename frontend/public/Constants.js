export class Constants {
    // TODO get from backend so it's all defined in one place
    static ROOM_SIZE_METERS_X = 40.0;
    static ROOM_SIZE_METERS_Y = 30.0;
    static PX_PER_METER = 20.0;

    static ROOM_SIZE_PIXELS_X = Constants.ROOM_SIZE_METERS_X * Constants.PX_PER_METER;
    static ROOM_SIZE_PIXELS_Y = Constants.ROOM_SIZE_METERS_Y * Constants.PX_PER_METER;

    static CANVAS_SIZE_X = Constants.ROOM_SIZE_PIXELS_X;
    static CANVAS_SIZE_Y = Constants.ROOM_SIZE_PIXELS_Y;

    static BULLET_SPEED_PX_SEC = 1024.0; // pixels per second
    static BULLET_RADIUS = 4;
    static BULLET_LIFETIME_SEC = 0.05;
    static BULLET_DESTROY_ANIMATION_TIME = 0.1;

    static PLAYER_DIED_ANIMATION_TIME_SEC = 3.0;
    static PLAYER_DIED_ANIMATION_BLINK_RATE_SEC = 0.2;
}