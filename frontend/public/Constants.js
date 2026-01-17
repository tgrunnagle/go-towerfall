export class Constants {
    static BULLET_SPEED_PX_SEC = 1024.0; // pixels per second
    static BULLET_RADIUS = 4;
    static BULLET_LIFETIME_SEC = 0.05;
    static BULLET_DESTROY_ANIMATION_TIME = 0.1;

    static PLAYER_DIED_ANIMATION_TIME_SEC = 3.0;
    static PLAYER_DIED_ANIMATION_BLINK_RATE_SEC = 0.2;

    static COLLISION_ANIMATION_TIME_SEC = 0.5;

    static SPECTATOR_TEXT_FONT = '20px Arial';
    static SPECTATOR_TEXT_COLOR = '#000000';
    static SPECTATOR_TEXT_OFFSET_X = 20;
    static SPECTATOR_TEXT_OFFSET_Y = 20;
    static SPECTATOR_TEXT_LINE_HEIGHT = 30;

    // Training mode overlay constants
    static TRAINING_TEXT_FONT = '16px Arial';
    static TRAINING_TEXT_COLOR = '#FFFFFF';
    static TRAINING_TEXT_BG_COLOR = 'rgba(0, 0, 0, 0.7)';
    static TRAINING_TEXT_PADDING = 10;
    static TRAINING_TEXT_LINE_HEIGHT = 22;
}