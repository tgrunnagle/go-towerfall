package constants

// Physics constants
const (
	AccelerationDueToGravity = 12.0 // 9.81
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
	ObjectTypeArrow  = "arrow"
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
	StateArrowGrounded     = "ag"   // arrow grounded state (boolean)
	StateShooting          = "sht"  // shooting state (boolean)
	StateShootingStartTime = "shts" // shooting start time (float64)
	StateJumpCount         = "jc"   // jumping count (int)
)

// Player constants
const (
	PlayerStartingX           = RoomSizePixelsX / 2.0
	PlayerStartingY          = RoomSizePixelsY / 2.0
	PlayerRadius             = 20.0
	PlayerSpeedXMetersPerSec = 10.0
	PlayerJumpSpeedMetersPerSec = 10.0
	PlayerStartingHealth     = 100
	MaxJumps                 = 2
	PlayerRespawnTimeSec     = 5.0
	PlayerMassKg             = 50.0
)

const (
	BulletDistance    = 1024.0 // Distance in pixels
	BulletLifetimeSec = 0.1
)

const (
	ArrowMaxPowerNewton        = 100.0
	ArrowMaxPowerTimeSec       = 2.0
	ArrowMassKg                = 0.1
	ArrowLengthMeters          = 1.0
	ArrowLengthPx              = ArrowLengthMeters * PxPerMeter
	ArrowDestroyDistanceMeters = 5.0
	ArrowDestroyDistancePx     = ArrowDestroyDistanceMeters * PxPerMeter
	ArrowGroundedRadiusMeters  = 0.5
	ArrowGroundedRadiusPx      = ArrowGroundedRadiusMeters * PxPerMeter
)
