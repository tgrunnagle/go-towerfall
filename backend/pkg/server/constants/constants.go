package constants

// Physics constants
const (
	AccelerationDueToGravityMetersPerSec2 = 20.0 // 9.81
	MaxVelocityMetersPerSec               = 30.0
	PxPerMeter                            = 20.0
)

// Room constants
const (
	RoomSizeMetersX        = 40.0
	RoomSizeMetersY        = 40.0
	RoomSizePixelsX        = RoomSizeMetersX * PxPerMeter
	RoomSizePixelsY        = RoomSizeMetersY * PxPerMeter
	RoomWrapDistanceMeters = 2.0
	RoomWrapDistancePx     = RoomWrapDistanceMeters * PxPerMeter
)

var RespawnLocationsPx = []struct {
	X float64
	Y float64
}{
	{X: 200, Y: 100},
	{X: 200, Y: 600},
	{X: 600, Y: 100},
	{X: 600, Y: 600},
}

// Object types
const (
	ObjectTypePlayer = "player"
	ObjectTypeBullet = "bullet"
	ObjectTypeBlock  = "block"
	ObjectTypeArrow  = "arrow"
)

// Object state keys
// Note these are sent to the client, so short names are preferred
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
	StatePoints            = "pts"  // array of points for polygon objects
	StateHealth            = "h"    // health of the object
	StateDestroyed         = "d"    // destroyed state (boolean)
	StateDestroyedAtX      = "dAtX" // x position when destroyed
	StateDestroyedAtY      = "dAtY" // y position when destroyed
	StateDead              = "dead" // dead state (boolean)
	StateArrowGrounded     = "ag"   // arrow grounded state (boolean)
	StateShooting          = "sht"  // shooting state (boolean)
	StateShootingStartTime = "shts" // shooting start time (float64)
	StateJumpCount         = "jc"   // jumping count (int)
	StateArrowCount        = "ac"   // arrow count (int)
)

// Player constants
const (
	PlayerRadius                = 20.0
	PlayerSpeedXMetersPerSec    = 15.0
	PlayerJumpSpeedMetersPerSec = 20.0
	PlayerStartingHealth        = 100
	PlayerMaxJumps              = 2
	PlayerRespawnTimeSec        = 5.0
	PlayerMassKg                = 50.0
	PlayerStartingArrows        = 4
	PlayerMaxArrows             = 4
)

// Bullet constants
const (
	BulletDistance    = 1024.0 // Distance in pixels
	BulletLifetimeSec = 0.1
)

// Arrow constants
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

// Block constants
const (
	BlockSizeUnitMeters float64 = 1.0
	BlockSizeUnitPixels float64 = BlockSizeUnitMeters * PxPerMeter
)
