package constants

// Physics constants
const (
	AccelerationDueToGravity = 9.81
	MaxVelocityMetersPerSec  = 20.0
	PxPerMeter               = 20.0
)

// Room constants
const (
	RoomSizeMetersX = 40.0
	RoomSizeMetersY = 30.0
	RoomSizePixelsX = RoomSizeMetersX * PxPerMeter
	RoomSizePixelsY = RoomSizeMetersY * PxPerMeter
)

// Object types
const (
	ObjectTypePlayer = "player"
	ObjectTypeBullet = "bullet"
	ObjectTypeBlock  = "block"
)

// Object state keys
const (
	StateID                = "id"
	StateName              = "name"
	StateX                 = "x"
	StateY                 = "y"
	StateWidth             = "w"
	StateHeight            = "h"
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
	PlayerRadius                = 16.0
	PlayerRespawnTimeSec        = 5.0
	PlayerSpeedXMetersPerSec    = 10.0
	PlayerJumpSpeedMetersPerSec = 20.0
	PlayerStartingX             = 100.0
	PlayerStartingY             = 100.0
	PlayerStartingHealth        = 100.0
)

const (
	BulletDistance    = 1024.0 // Distance in pixels
	BulletLifetimeSec = 0.1
)
