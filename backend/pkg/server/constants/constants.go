package constants

// Room constants
const (
	RoomSizeX = 800
	RoomSizeY = 600
)

// Object types
const (
	ObjectTypePlayer = "player"
	ObjectTypeBullet = "bullet"
)

// Object state keys
const (
	StateID                = "id"
	StateName              = "name"
	StateX                 = "x"
	StateY                 = "y"
	StateDx                = "dx"
	StateDy                = "dy"
	StateLastLocUpdateTime = "llut"
	StateDir               = "dir"  // direction in radians, 0 is right
	StateRadius            = "rad"  // radius of the object bounding circle
	StateHealth            = "h"    // health of the object
	StateDestroyed         = "d"    // destroyed state (boolean)
	StateDestroyedAtX      = "dAtX" // x position when destroyed
	StateDestroyedAtY      = "dAtY" // y position when destroyed
	StateDead              = "dead" // dead state (boolean)
)

// Object constants
const (
	PlayerRadius         = 16.0
	PlayerRespawnTimeSec = 5.0
)

const (
	BulletDistance    = 1024.0 // Distance in pixels
	BulletLifetimeSec = 0.1
)
