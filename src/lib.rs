pub use coord_2d::Coord;
pub use direction::Direction;
use direction::{CardinalDirection, OrdinalDirection};
use either::Either;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::iter;

fn delta_num_steps(delta: Coord) -> u32 {
    delta.x.abs().max(delta.y.abs()) as u32 + 1
}

fn delta_num_cardinal_steps(delta: Coord) -> u32 {
    delta.x.abs() as u32 + delta.y.abs() as u32 + 1
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StepsCommon {
    major: Direction,
    minor: Direction,
    accumulator: i64,
    major_delta_abs: u32,
    minor_delta_abs: u32,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Eq, Debug)]
struct Steps(StepsCommon);

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Eq, Debug)]
struct CardinalSteps(StepsCommon);

struct StepsDesc {
    major: CardinalDirection,
    minor: CardinalDirection,
    major_delta_abs: u32,
    minor_delta_abs: u32,
}

impl StepsDesc {
    fn new(delta: Coord) -> Self {
        let delta_x_abs = delta.x.abs() as u32;
        let delta_y_abs = delta.y.abs() as u32;
        if delta_x_abs > delta_y_abs {
            let major = if delta.x > 0 {
                CardinalDirection::East
            } else {
                CardinalDirection::West
            };
            let minor = if delta.y > 0 {
                CardinalDirection::South
            } else {
                CardinalDirection::North
            };
            Self {
                major,
                minor,
                major_delta_abs: delta_x_abs,
                minor_delta_abs: delta_y_abs,
            }
        } else {
            let major = if delta.y > 0 {
                CardinalDirection::South
            } else {
                CardinalDirection::North
            };
            let minor = if delta.x > 0 {
                CardinalDirection::East
            } else {
                CardinalDirection::West
            };
            Self {
                major,
                minor,
                major_delta_abs: delta_y_abs,
                minor_delta_abs: delta_x_abs,
            }
        }
    }
    fn into_steps(self) -> Steps {
        let Self {
            major,
            minor,
            major_delta_abs,
            minor_delta_abs,
        } = self;
        Steps(StepsCommon {
            major: major.direction(),
            minor: OrdinalDirection::from_cardinals(major, minor)
                .unwrap()
                .direction(),
            accumulator: 0,
            major_delta_abs,
            minor_delta_abs,
        })
    }
    fn into_cardinal_steps(self) -> CardinalSteps {
        let Self {
            major,
            minor,
            major_delta_abs,
            minor_delta_abs,
        } = self;
        CardinalSteps(StepsCommon {
            major: major.direction(),
            minor: minor.direction(),
            accumulator: 0,
            major_delta_abs,
            minor_delta_abs,
        })
    }
}

trait StepsTrait: Clone {
    fn prev(&mut self) -> Direction;
    fn next(&mut self) -> Direction;
}

impl StepsTrait for Steps {
    fn prev(&mut self) -> Direction {
        self.0.accumulator -= self.0.minor_delta_abs as i64;
        if self.0.accumulator <= (self.0.major_delta_abs as i64 / 2) - self.0.major_delta_abs as i64
        {
            self.0.accumulator += self.0.major_delta_abs as i64;
            self.0.minor.opposite()
        } else {
            self.0.major.opposite()
        }
    }
    fn next(&mut self) -> Direction {
        self.0.accumulator += self.0.minor_delta_abs as i64;
        if self.0.accumulator > self.0.major_delta_abs as i64 / 2 {
            self.0.accumulator -= self.0.major_delta_abs as i64;
            self.0.minor
        } else {
            self.0.major
        }
    }
}

impl StepsTrait for CardinalSteps {
    fn prev(&mut self) -> Direction {
        self.0.accumulator -= self.0.minor_delta_abs as i64;
        if self.0.accumulator
            <= (self.0.major_delta_abs as i64 / 2)
                - self.0.major_delta_abs as i64
                - self.0.minor_delta_abs as i64
        {
            self.0.accumulator += self.0.major_delta_abs as i64 + self.0.minor_delta_abs as i64;
            self.0.minor.opposite()
        } else {
            self.0.major.opposite()
        }
    }
    fn next(&mut self) -> Direction {
        self.0.accumulator += self.0.minor_delta_abs as i64;
        if self.0.accumulator > self.0.major_delta_abs as i64 / 2 {
            self.0.accumulator -= self.0.major_delta_abs as i64 + self.0.minor_delta_abs as i64;
            self.0.minor
        } else {
            self.0.major
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
struct GeneralInfiniteIter<S>
where
    S: StepsTrait,
{
    steps: S,
    current: Coord,
}

impl<S> GeneralInfiniteIter<S>
where
    S: StepsTrait,
{
    fn new_exclude_start(start: Coord, steps: S) -> Self {
        Self {
            current: start,
            steps,
        }
    }
    fn new_include_start(start: Coord, steps: S) -> Self {
        let mut iter = Self::new_exclude_start(start, steps);
        let backward_step = iter.steps.prev();
        iter.current += backward_step.coord();
        iter
    }
    fn into_infinite_node_iter(self) -> GeneralInfiniteNodeIter<S> {
        GeneralInfiniteNodeIter {
            current: self.current,
            steps: self.steps,
        }
    }
}

impl<S> Iterator for GeneralInfiniteIter<S>
where
    S: StepsTrait,
{
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        let step = self.steps.next().coord();
        if let Some(next_coord) = self.current.checked_add(step) {
            self.current = next_coord;
            Some(next_coord)
        } else {
            None
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InfiniteIter(GeneralInfiniteIter<Steps>);

impl Iterator for InfiniteIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InfiniteCardinalIter(GeneralInfiniteIter<CardinalSteps>);

impl Iterator for InfiniteCardinalIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
struct Finite<I> {
    iter: I,
    remaining: u32,
}

impl<I> Iterator for Finite<I>
where
    I: Iterator,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.iter.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Iter(Finite<InfiniteIter>);

impl Iterator for Iter {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct CardinalIter(Finite<InfiniteCardinalIter>);

impl Iterator for CardinalIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[derive(Clone, Copy)]
pub struct Node {
    pub next: Direction,
    pub prev: Direction,
    pub coord: Coord,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
struct GeneralInfiniteNodeIter<S>
where
    S: StepsTrait,
{
    steps: S,
    current: Coord,
}

impl<S> Iterator for GeneralInfiniteNodeIter<S>
where
    S: StepsTrait,
{
    type Item = Node;
    fn next(&mut self) -> Option<Self::Item> {
        let step_direction = self.steps.next();
        let step = step_direction.coord();
        let prev = step_direction.opposite();
        let next = self.steps.clone().next();
        if let Some(next_coord) = self.current.checked_add(step) {
            self.current = next_coord;
            Some(Node {
                coord: next_coord,
                next,
                prev,
            })
        } else {
            None
        }
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InfiniteNodeIter(GeneralInfiniteNodeIter<Steps>);

impl Iterator for InfiniteNodeIter {
    type Item = Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InfiniteCardinalNodeIter(GeneralInfiniteNodeIter<CardinalSteps>);

impl Iterator for InfiniteCardinalNodeIter {
    type Item = Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct NodeIter(Finite<InfiniteNodeIter>);

impl Iterator for NodeIter {
    type Item = Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct CardinalNodeIter(Finite<InfiniteCardinalNodeIter>);

impl Iterator for CardinalNodeIter {
    type Item = Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[derive(Clone, Copy)]
pub struct Config {
    pub exclude_start: bool,
    pub exclude_end: bool,
}

#[derive(Clone, Copy)]
pub struct InfiniteConfig {
    pub exclude_start: bool,
}

impl Config {
    fn into_infinite(self) -> InfiniteConfig {
        InfiniteConfig {
            exclude_start: self.exclude_start,
        }
    }
}

/// A straight line between two different points
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineSegment {
    start: Coord,
    end: Coord,
}

#[derive(Debug)]
pub struct StartAndEndAreTheSame;

#[derive(Debug)]
pub struct ZeroDelta;

impl LineSegment {
    pub fn try_new(start: Coord, end: Coord) -> Result<Self, StartAndEndAreTheSame> {
        if start == end {
            Err(StartAndEndAreTheSame)
        } else {
            Ok(Self { start, end })
        }
    }

    /// Creates a new `LineSegment` panicking if `start` and `end` are the same
    pub fn new(start: Coord, end: Coord) -> Self {
        if start == end {
            panic!("start and end must be different");
        }
        Self { start, end }
    }
    pub fn start(&self) -> Coord {
        self.start
    }
    pub fn end(&self) -> Coord {
        self.end
    }
    pub fn delta(&self) -> Coord {
        self.end - self.start
    }
    pub fn num_steps(&self) -> u32 {
        delta_num_steps(self.delta())
    }
    pub fn num_cardinal_steps(&self) -> u32 {
        delta_num_cardinal_steps(self.delta())
    }
    pub fn reverse(&self) -> Self {
        Self {
            start: self.end,
            end: self.start,
        }
    }
    fn steps(&self) -> Steps {
        StepsDesc::new(self.delta()).into_steps()
    }
    fn cardinal_steps(&self) -> CardinalSteps {
        StepsDesc::new(self.delta()).into_cardinal_steps()
    }
    pub fn infinite_iter(&self) -> InfiniteIter {
        InfiniteIter(GeneralInfiniteIter::new_include_start(
            self.start,
            self.steps(),
        ))
    }
    pub fn infinite_cardinal_iter(&self) -> InfiniteCardinalIter {
        InfiniteCardinalIter(GeneralInfiniteIter::new_include_start(
            self.start,
            self.cardinal_steps(),
        ))
    }
    pub fn config_infinite_iter(&self, config: InfiniteConfig) -> InfiniteIter {
        if config.exclude_start {
            InfiniteIter(GeneralInfiniteIter::new_exclude_start(
                self.start,
                self.steps(),
            ))
        } else {
            InfiniteIter(GeneralInfiniteIter::new_include_start(
                self.start,
                self.steps(),
            ))
        }
    }
    pub fn config_infinite_cardinal_iter(&self, config: InfiniteConfig) -> InfiniteCardinalIter {
        if config.exclude_start {
            InfiniteCardinalIter(GeneralInfiniteIter::new_exclude_start(
                self.start,
                self.cardinal_steps(),
            ))
        } else {
            InfiniteCardinalIter(GeneralInfiniteIter::new_include_start(
                self.start,
                self.cardinal_steps(),
            ))
        }
    }

    /// Iterator over all coordinates allowing ordinal steps which begins on the start coordinate
    /// and ends with the end coordinate (inclusively)
    pub fn iter(&self) -> Iter {
        Iter(Finite {
            iter: self.infinite_iter(),
            remaining: self.num_steps(),
        })
    }

    /// Iterator over all coordinates allowing only cardinal steps which begins on the start
    /// coordinate and ends with the end coordinate (inclusively)
    pub fn cardinal_iter(&self) -> CardinalIter {
        CardinalIter(Finite {
            iter: self.infinite_cardinal_iter(),
            remaining: self.num_cardinal_steps(),
        })
    }
    pub fn config_iter(&self, config: Config) -> Iter {
        let iter = self.config_infinite_iter(config.into_infinite());
        let remaining = if let Some(num_steps) = self
            .num_steps()
            .checked_sub(config.exclude_start as u32 + config.exclude_end as u32)
        {
            num_steps
        } else {
            0
        };
        Iter(Finite { iter, remaining })
    }
    pub fn config_cardinal_iter(&self, config: Config) -> CardinalIter {
        let iter = self.config_infinite_cardinal_iter(config.into_infinite());
        let remaining = if let Some(num_steps) = self
            .num_cardinal_steps()
            .checked_sub(config.exclude_start as u32 + config.exclude_end as u32)
        {
            num_steps
        } else {
            0
        };
        CardinalIter(Finite { iter, remaining })
    }

    pub fn infinite_node_iter(&self) -> InfiniteNodeIter {
        InfiniteNodeIter(
            GeneralInfiniteIter::new_include_start(self.start, self.steps())
                .into_infinite_node_iter(),
        )
    }
    pub fn infinite_cardinal_node_iter(&self) -> InfiniteCardinalNodeIter {
        InfiniteCardinalNodeIter(
            GeneralInfiniteIter::new_include_start(self.start, self.cardinal_steps())
                .into_infinite_node_iter(),
        )
    }
    pub fn config_infinite_node_iter(&self, config: InfiniteConfig) -> InfiniteNodeIter {
        if config.exclude_start {
            InfiniteNodeIter(
                GeneralInfiniteIter::new_exclude_start(self.start, self.steps())
                    .into_infinite_node_iter(),
            )
        } else {
            InfiniteNodeIter(
                GeneralInfiniteIter::new_include_start(self.start, self.steps())
                    .into_infinite_node_iter(),
            )
        }
    }
    pub fn config_infinite_cardinal_node_iter(
        &self,
        config: InfiniteConfig,
    ) -> InfiniteCardinalNodeIter {
        if config.exclude_start {
            InfiniteCardinalNodeIter(
                GeneralInfiniteIter::new_exclude_start(self.start, self.cardinal_steps())
                    .into_infinite_node_iter(),
            )
        } else {
            InfiniteCardinalNodeIter(
                GeneralInfiniteIter::new_include_start(self.start, self.cardinal_steps())
                    .into_infinite_node_iter(),
            )
        }
    }
    pub fn node_iter(&self) -> NodeIter {
        NodeIter(Finite {
            iter: self.infinite_node_iter(),
            remaining: self.num_steps(),
        })
    }
    pub fn cardinal_node_iter(&self) -> CardinalNodeIter {
        CardinalNodeIter(Finite {
            iter: self.infinite_cardinal_node_iter(),
            remaining: self.num_cardinal_steps(),
        })
    }
    pub fn config_node_iter(&self, config: Config) -> NodeIter {
        let iter = self.config_infinite_node_iter(config.into_infinite());
        let remaining = if let Some(num_steps) = self
            .num_steps()
            .checked_sub(config.exclude_start as u32 + config.exclude_end as u32)
        {
            num_steps
        } else {
            0
        };
        NodeIter(Finite { iter, remaining })
    }
    pub fn config_cardinal_node_iter(&self, config: Config) -> CardinalNodeIter {
        let iter = self.config_infinite_cardinal_node_iter(config.into_infinite());
        let remaining = if let Some(num_steps) = self
            .num_cardinal_steps()
            .checked_sub(config.exclude_start as u32 + config.exclude_end as u32)
        {
            num_steps
        } else {
            0
        };
        CardinalNodeIter(Finite { iter, remaining })
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InfiniteStepIter(Steps);

impl InfiniteStepIter {
    pub fn try_new(delta: Coord) -> Result<Self, ZeroDelta> {
        if delta == Coord::new(0, 0) {
            Err(ZeroDelta)
        } else {
            Ok(Self(StepsDesc::new(delta).into_steps()))
        }
    }
    pub fn new(delta: Coord) -> Self {
        if delta == Coord::new(0, 0) {
            panic!("delta must not be zero");
        } else {
            Self(StepsDesc::new(delta).into_steps())
        }
    }
    pub fn step(&mut self) -> Direction {
        self.0.next()
    }
    pub fn step_back(&mut self) -> Direction {
        self.0.prev()
    }
}

impl Iterator for InfiniteStepIter {
    type Item = Direction;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next())
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InfiniteCardinalStepIter(CardinalSteps);

impl InfiniteCardinalStepIter {
    pub fn try_new(delta: Coord) -> Result<Self, ZeroDelta> {
        if delta == Coord::new(0, 0) {
            Err(ZeroDelta)
        } else {
            Ok(Self(StepsDesc::new(delta).into_cardinal_steps()))
        }
    }
    pub fn new(delta: Coord) -> Self {
        if delta == Coord::new(0, 0) {
            panic!("delta must not be zero");
        } else {
            Self(StepsDesc::new(delta).into_cardinal_steps())
        }
    }
    pub fn step(&mut self) -> Direction {
        self.0.next()
    }
    pub fn step_back(&mut self) -> Direction {
        self.0.prev()
    }
}

impl Iterator for InfiniteCardinalStepIter {
    type Item = Direction;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next())
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct StepIter {
    infinite: InfiniteStepIter,
    remaining: u32,
}

impl StepIter {
    pub fn try_new(delta: Coord) -> Result<Self, ZeroDelta> {
        Ok(Self {
            infinite: InfiniteStepIter::try_new(delta)?,
            remaining: delta_num_steps(delta),
        })
    }
    pub fn new(delta: Coord) -> Self {
        Self {
            infinite: InfiniteStepIter::new(delta),
            remaining: delta_num_steps(delta),
        }
    }
}

impl Iterator for StepIter {
    type Item = Direction;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.infinite.next()
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct CardinalStepIter {
    infinite: InfiniteCardinalStepIter,
    remaining: u32,
}

impl CardinalStepIter {
    pub fn try_new(delta: Coord) -> Result<Self, ZeroDelta> {
        Ok(Self {
            infinite: InfiniteCardinalStepIter::try_new(delta)?,
            remaining: delta_num_cardinal_steps(delta),
        })
    }
    pub fn new(delta: Coord) -> Self {
        Self {
            infinite: InfiniteCardinalStepIter::new(delta),
            remaining: delta_num_cardinal_steps(delta),
        }
    }
}

impl Iterator for CardinalStepIter {
    type Item = Direction;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.infinite.next()
    }
}

/// The `LineSegment` type can't represent line segments of 0 length. It's often convenient for
/// users to treat 0-length line segments as points. This type represent line segments which can be
/// of 0 length, treating such lines as single points.
pub enum LineSegmentOrPoint {
    LineSegment(LineSegment),
    Point(Coord),
}

pub type LineSegmentOrPointIter = Either<Iter, iter::Once<Coord>>;
pub type LineSegmentOrPointCardinalIter = Either<CardinalIter, iter::Once<Coord>>;

impl LineSegmentOrPoint {
    pub fn new(start: Coord, end: Coord) -> Self {
        match LineSegment::try_new(start, end) {
            Ok(line_segment) => Self::LineSegment(line_segment),
            Err(StartAndEndAreTheSame) => Self::Point(start),
        }
    }
    pub fn start(&self) -> Coord {
        match self {
            Self::LineSegment(line_segment) => line_segment.start(),
            Self::Point(coord) => *coord,
        }
    }
    pub fn end(&self) -> Coord {
        match self {
            Self::LineSegment(line_segment) => line_segment.end(),
            Self::Point(coord) => *coord,
        }
    }
    pub fn delta(&self) -> Coord {
        match self {
            Self::LineSegment(line_segment) => line_segment.delta(),
            Self::Point(_) => Coord::new(0, 0),
        }
    }
    pub fn num_steps(&self) -> u32 {
        match self {
            Self::LineSegment(line_segment) => line_segment.num_steps(),
            Self::Point(_) => 0,
        }
    }
    pub fn num_cardinal_steps(&self) -> u32 {
        match self {
            Self::LineSegment(line_segment) => line_segment.num_cardinal_steps(),
            Self::Point(_) => 0,
        }
    }
    pub fn reverse(&self) -> Self {
        match self {
            Self::LineSegment(line_segment) => Self::LineSegment(line_segment.reverse()),
            Self::Point(coord) => Self::Point(*coord),
        }
    }

    /// Iterator over all coordinates allowing ordinal steps which begins on the start coordinate
    /// and ends with the end coordinate (inclusively)
    pub fn iter(&self) -> LineSegmentOrPointIter {
        match self {
            Self::LineSegment(line_segment) => Either::Left(line_segment.iter()),
            Self::Point(coord) => Either::Right(iter::once(*coord)),
        }
    }

    /// Iterator over all coordinates allowing only cardinal steps which begins on the start
    /// coordinate and ends with the end coordinate (inclusively)
    pub fn cardinal_iter(&self) -> LineSegmentOrPointCardinalIter {
        match self {
            Self::LineSegment(line_segment) => Either::Left(line_segment.cardinal_iter()),
            Self::Point(coord) => Either::Right(iter::once(*coord)),
        }
    }
}

/// Returns an iterator over all the coordinates between start and end (inclusive) along a straight
/// (rasterized) line which includes ordinal steps. `start` and `end` may be the same.
pub fn coords_between(start: Coord, end: Coord) -> impl Iterator<Item = Coord> {
    LineSegmentOrPoint::new(start, end).iter()
}

/// Returns an iterator over all the coordinates between start and end (inclusive) along a straight
/// (rasterized) line which includes only cardinal steps. `start` and `end` may be the same.
pub fn coords_between_cardinal(start: Coord, end: Coord) -> impl Iterator<Item = Coord> {
    LineSegmentOrPoint::new(start, end).cardinal_iter()
}

#[cfg(test)]
mod test {
    use super::*;
    use grid_2d::{Grid, Size};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    trait TraverseTrait {
        type Steps: StepsTrait + Clone + Eq;
        type Iter: Iterator<Item = Coord> + ::std::fmt::Debug;
        fn iter(&self) -> Self::Iter;
        fn config_iter(&self, config: Config) -> Self::Iter;
        fn num_steps(&self) -> u32;
        fn start(&self) -> Coord;
        fn end(&self) -> Coord;
        fn steps(&self) -> Self::Steps;
    }

    #[derive(Debug)]
    struct Traverse(LineSegment);

    #[derive(Debug)]
    struct CardinalTraverse(LineSegment);

    impl TraverseTrait for Traverse {
        type Steps = Steps;
        type Iter = Iter;
        fn iter(&self) -> Self::Iter {
            self.0.iter()
        }
        fn config_iter(&self, config: Config) -> Self::Iter {
            self.0.config_iter(config)
        }
        fn num_steps(&self) -> u32 {
            self.0.num_steps()
        }
        fn start(&self) -> Coord {
            self.0.start
        }
        fn end(&self) -> Coord {
            self.0.end
        }
        fn steps(&self) -> Self::Steps {
            self.0.steps()
        }
    }

    impl TraverseTrait for CardinalTraverse {
        type Steps = CardinalSteps;
        type Iter = CardinalIter;
        fn iter(&self) -> Self::Iter {
            self.0.cardinal_iter()
        }
        fn config_iter(&self, config: Config) -> Self::Iter {
            self.0.config_cardinal_iter(config)
        }
        fn num_steps(&self) -> u32 {
            self.0.num_cardinal_steps()
        }
        fn start(&self) -> Coord {
            self.0.start
        }
        fn end(&self) -> Coord {
            self.0.end
        }
        fn steps(&self) -> Self::Steps {
            self.0.cardinal_steps()
        }
    }

    trait InfiniteTraverseTrait {
        type Iter: Iterator<Item = Coord>;
        fn iter(&self) -> Self::Iter;
        fn config_iter(&self, config: InfiniteConfig) -> Self::Iter;
    }

    struct InfiniteTraverse(LineSegment);
    impl InfiniteTraverseTrait for InfiniteTraverse {
        type Iter = InfiniteIter;
        fn iter(&self) -> Self::Iter {
            self.0.infinite_iter()
        }
        fn config_iter(&self, config: InfiniteConfig) -> Self::Iter {
            self.0.config_infinite_iter(config)
        }
    }

    struct InfiniteCardinalTraverse(LineSegment);
    impl InfiniteTraverseTrait for InfiniteCardinalTraverse {
        type Iter = InfiniteCardinalIter;
        fn iter(&self) -> Self::Iter {
            self.0.infinite_cardinal_iter()
        }
        fn config_iter(&self, config: InfiniteConfig) -> Self::Iter {
            self.0.config_infinite_cardinal_iter(config)
        }
    }

    fn test_properties_gen<T>(traverse: T)
    where
        T: TraverseTrait + ::std::fmt::Debug,
        T::Steps: ::std::fmt::Debug,
    {
        let coords: Vec<_> = traverse.iter().collect();
        assert_eq!(coords.len() as u32, traverse.num_steps());
        assert_eq!(*coords.first().unwrap(), traverse.start());
        assert_eq!(*coords.last().unwrap(), traverse.end());
        let mut steps = traverse.steps();
        for _ in 0..traverse.num_steps() {
            let before = steps.clone();
            steps.next();
            let mut after = steps.clone();
            after.prev();
            assert_eq!(
                before, after,
                "\n{:#?}\n{:#?}\n{:#?}",
                before, after, traverse
            );
        }
        let orig_coords = coords;
        let coords: Vec<_> = traverse
            .config_iter(Config {
                exclude_start: true,
                exclude_end: true,
            })
            .collect();
        assert_eq!(coords.len() as u32, traverse.num_steps().max(2) - 2);
        if let Some(&coord) = coords.first() {
            assert_eq!(coord, orig_coords[1]);
        }
        if let Some(&coord) = coords.last() {
            assert_eq!(coord, orig_coords[orig_coords.len() - 2]);
        }
        let coords: Vec<_> = traverse
            .config_iter(Config {
                exclude_start: true,
                exclude_end: false,
            })
            .collect();
        assert_eq!(coords.len() as u32, traverse.num_steps().max(1) - 1);
        if let Some(&coord) = coords.first() {
            assert_eq!(coord, orig_coords[1]);
        }
        if let Some(&coord) = coords.last() {
            assert_eq!(coord, orig_coords[orig_coords.len() - 1]);
        }
        let coords: Vec<_> = traverse
            .config_iter(Config {
                exclude_start: false,
                exclude_end: true,
            })
            .collect();
        assert_eq!(coords.len() as u32, traverse.num_steps().max(1) - 1);
        if let Some(&coord) = coords.first() {
            assert_eq!(coord, orig_coords[0]);
        }
        if let Some(&coord) = coords.last() {
            assert_eq!(coord, orig_coords[orig_coords.len() - 2]);
        }
    }

    fn compare_finite_with_infinite<F: TraverseTrait, I: InfiniteTraverseTrait>(f: F, i: I) {
        f.iter().zip(i.iter()).for_each(|(f, i)| {
            assert_eq!(f, i);
        });
        f.config_iter(Config {
            exclude_start: true,
            exclude_end: false,
        })
        .zip(i.config_iter(InfiniteConfig {
            exclude_start: true,
        }))
        .for_each(|(f, i)| {
            assert_eq!(f, i);
        });
    }

    fn test_properties(line_segment: LineSegment) {
        test_properties_gen(Traverse(line_segment));
        test_properties_gen(CardinalTraverse(line_segment));
        compare_finite_with_infinite(Traverse(line_segment), InfiniteTraverse(line_segment));
        compare_finite_with_infinite(
            CardinalTraverse(line_segment),
            InfiniteCardinalTraverse(line_segment),
        );
    }

    fn rand_int<R: Rng>(rng: &mut R) -> i32 {
        const MAX: i32 = 100;
        rng.gen::<i32>() % MAX
    }

    fn rand_coord<R: Rng>(rng: &mut R) -> Coord {
        Coord::new(rand_int(rng), rand_int(rng))
    }

    fn rand_line_segment<R: Rng>(rng: &mut R) -> LineSegment {
        let start = rand_coord(rng);
        let mut end = rand_coord(rng);
        if start == end {
            end.x += 1;
        }
        LineSegment::new(start, end)
    }

    #[test]
    fn all() {
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(1, 1)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(1, 0)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(2, 1)));
        test_properties(LineSegment::new(Coord::new(1, -1), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(1, 100), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(100, 1), Coord::new(0, 0)));
        const NUM_RAND_TESTS: u32 = 10000;
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..NUM_RAND_TESTS {
            test_properties(rand_line_segment(&mut rng));
        }
    }

    #[test]
    fn infinite_traverse() {
        let line = LineSegment::new(Coord::new(0, 0), Coord::new(1, 0));
        let first_4 = InfiniteTraverse(line).iter().take(4).collect::<Vec<_>>();
        assert_eq!(
            &first_4,
            &[
                Coord::new(0, 0),
                Coord::new(1, 0),
                Coord::new(2, 0),
                Coord::new(3, 0),
            ]
        );
    }

    fn render_iter<I>(iter: I) -> Vec<String>
    where
        I: Iterator<Item = Coord>,
    {
        let mut grid = Grid::new_clone(Size::new(10, 10), '.');
        for coord in iter {
            *grid.get_checked_mut(coord) = '#';
        }
        let mut v = Vec::new();
        for row in grid.rows() {
            let mut s = String::new();
            for &cell in row {
                s.push(cell);
            }
            v.push(s);
        }
        println!("{:?}", v);
        v
    }

    fn render_nodes<I>(iter: I) -> Vec<String>
    where
        I: Iterator<Item = Node>,
    {
        fn render_node(node: Node) -> char {
            match node.next {
                Direction::North | Direction::South => '|',
                Direction::East | Direction::West => '-',
                Direction::NorthEast | Direction::SouthWest => '/',
                Direction::NorthWest | Direction::SouthEast => '\\',
            }
        }
        let mut grid = Grid::new_clone(Size::new(10, 10), '.');
        for node in iter {
            let ch = render_node(node);
            *grid.get_checked_mut(node.coord) = ch;
        }
        let mut v = Vec::new();
        for row in grid.rows() {
            let mut s = String::new();
            for &cell in row {
                s.push(cell);
            }
            v.push(s);
        }
        println!("{:?}", v);
        v
    }

    #[test]
    fn examples() {
        assert_eq!(
            render_iter(LineSegment::new(Coord::new(2, 3), Coord::new(8, 6)).iter()),
            &[
                "..........",
                "..........",
                "..........",
                "..##......",
                "....##....",
                "......##..",
                "........#.",
                "..........",
                "..........",
                ".........."
            ]
        );
        assert_eq!(
            render_iter(
                LineSegment::new(Coord::new(2, 3), Coord::new(8, 6)).config_iter(Config {
                    exclude_start: true,
                    exclude_end: true
                })
            ),
            &[
                "..........",
                "..........",
                "..........",
                "...#......",
                "....##....",
                "......##..",
                "..........",
                "..........",
                "..........",
                ".........."
            ]
        );

        assert_eq!(
            render_iter(LineSegment::new(Coord::new(2, 3), Coord::new(8, 6)).cardinal_iter()),
            &[
                "..........",
                "..........",
                "..........",
                "..##......",
                "...###....",
                ".....###..",
                ".......##.",
                "..........",
                "..........",
                ".........."
            ]
        );
        assert_eq!(
            render_iter(LineSegment::new(Coord::new(6, 2), Coord::new(6, 7)).iter()),
            &[
                "..........",
                "..........",
                "......#...",
                "......#...",
                "......#...",
                "......#...",
                "......#...",
                "......#...",
                "..........",
                ".........."
            ]
        );
        assert_eq!(
            render_iter(LineSegment::new(Coord::new(3, 7), Coord::new(8, 7)).iter()),
            &[
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "..........",
                "...######.",
                "..........",
                ".........."
            ]
        );
        assert_eq!(
            render_nodes(LineSegment::new(Coord::new(7, 1), Coord::new(2, 8)).node_iter()),
            &[
                "..........",
                "......./..",
                "......|...",
                "....../...",
                "...../....",
                "..../.....",
                "...|......",
                ".../......",
                "../.......",
                ".........."
            ]
        );
    }

    #[test]
    fn step() {
        use Direction::*;
        assert_eq!(
            StepIter::new(Coord::new(4, 1)).collect::<Vec<_>>(),
            [East, East, SouthEast, East, East]
        );
        assert_eq!(
            CardinalStepIter::new(Coord::new(3, 1)).collect::<Vec<_>>(),
            [East, South, East, East, East]
        );
    }

    #[test]
    fn basic() {
        let s = LineSegment::new(Coord::new(1, 1), Coord::new(3, 1))
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(
            s,
            vec![Coord::new(1, 1), Coord::new(2, 1), Coord::new(3, 1)]
        );
    }
}
