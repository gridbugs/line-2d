#[cfg(feature = "serialize")]
#[macro_use]
extern crate serde;
extern crate coord_2d;
extern crate direction;
pub use coord_2d::Coord;
pub use direction::{CardinalDirection, Direction, OrdinalDirection};
use std::iter::Take;

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
    fn prev(&mut self) -> Coord;
    fn next(&mut self) -> Coord;
}

impl StepsTrait for Steps {
    fn prev(&mut self) -> Coord {
        if self.0.major_delta_abs == 0 {
            return Coord::new(0, 0);
        }
        self.0.accumulator -= self.0.minor_delta_abs as i64;
        if self.0.accumulator <= (self.0.major_delta_abs as i64 / 2) - self.0.major_delta_abs as i64
        {
            self.0.accumulator += self.0.major_delta_abs as i64;
            self.0.minor.opposite().coord()
        } else {
            self.0.major.opposite().coord()
        }
    }
    fn next(&mut self) -> Coord {
        if self.0.major_delta_abs == 0 {
            return Coord::new(0, 0);
        }
        self.0.accumulator += self.0.minor_delta_abs as i64;
        if self.0.accumulator > self.0.major_delta_abs as i64 / 2 {
            self.0.accumulator -= self.0.major_delta_abs as i64;
            self.0.minor.coord()
        } else {
            self.0.major.coord()
        }
    }
}

impl StepsTrait for CardinalSteps {
    fn prev(&mut self) -> Coord {
        if self.0.major_delta_abs == 0 {
            return Coord::new(0, 0);
        }
        self.0.accumulator -= self.0.minor_delta_abs as i64;
        if self.0.accumulator
            <= (self.0.major_delta_abs as i64 / 2)
                - self.0.major_delta_abs as i64
                - self.0.minor_delta_abs as i64
        {
            self.0.accumulator += self.0.major_delta_abs as i64 + self.0.minor_delta_abs as i64;;
            self.0.minor.opposite().coord()
        } else {
            self.0.major.opposite().coord()
        }
    }
    fn next(&mut self) -> Coord {
        if self.0.major_delta_abs == 0 {
            return Coord::new(0, 0);
        }
        self.0.accumulator += self.0.minor_delta_abs as i64;
        if self.0.accumulator > self.0.major_delta_abs as i64 / 2 {
            self.0.accumulator -= self.0.major_delta_abs as i64 + self.0.minor_delta_abs as i64;
            self.0.minor.coord()
        } else {
            self.0.major.coord()
        }
    }
}

#[derive(Debug)]
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
        iter.current += backward_step;
        iter
    }
}

impl<S> Iterator for GeneralInfiniteIter<S>
where
    S: StepsTrait,
{
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        let step = self.steps.next();
        if let Some(next_coord) = self.current.checked_add(step) {
            self.current = next_coord;
            Some(next_coord)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct InfiniteIter(GeneralInfiniteIter<Steps>);

impl Iterator for InfiniteIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[derive(Debug)]
pub struct InfiniteCardinalIter(GeneralInfiniteIter<CardinalSteps>);

impl Iterator for InfiniteCardinalIter {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub type Iter = Take<InfiniteIter>;
pub type CardinalIter = Take<InfiniteCardinalIter>;

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

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineSegment {
    pub start: Coord,
    pub end: Coord,
}

impl LineSegment {
    pub fn new(start: Coord, end: Coord) -> Self {
        Self { start, end }
    }
    pub fn delta(&self) -> Coord {
        self.end - self.start
    }
    pub fn num_steps(&self) -> usize {
        let delta = self.delta();
        delta.x.abs().max(delta.y.abs()) as usize + 1
    }
    pub fn num_cardinal_steps(&self) -> usize {
        let delta = self.delta();
        delta.x.abs() as usize + delta.y.abs() as usize + 1
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
    pub fn iter(&self) -> Iter {
        self.infinite_iter().take(self.num_steps())
    }
    pub fn cardinal_iter(&self) -> CardinalIter {
        self.infinite_cardinal_iter()
            .take(self.num_cardinal_steps())
    }
    pub fn config_iter(&self, config: Config) -> Iter {
        let infinite = self.config_infinite_iter(config.into_infinite());
        if let Some(num_steps) = self
            .num_steps()
            .checked_sub(config.exclude_start as usize + config.exclude_end as usize)
        {
            infinite.take(num_steps)
        } else {
            infinite.take(0)
        }
    }
    pub fn config_cardinal_iter(&self, config: Config) -> CardinalIter {
        let infinite = self.config_infinite_cardinal_iter(config.into_infinite());
        if let Some(num_steps) = self
            .num_cardinal_steps()
            .checked_sub(config.exclude_start as usize + config.exclude_end as usize)
        {
            infinite.take(num_steps)
        } else {
            infinite.take(0)
        }
    }
}

#[cfg(test)]
mod test {
    extern crate grid_2d;
    extern crate rand;
    use self::grid_2d::{Grid, Size};
    use self::rand::rngs::StdRng;
    use self::rand::{Rng, SeedableRng};
    use super::*;

    trait TraverseTrait {
        type Steps: StepsTrait + Clone + Eq;
        type Iter: Iterator<Item = Coord> + ::std::fmt::Debug;
        fn iter(&self) -> Self::Iter;
        fn config_iter(&self, config: Config) -> Self::Iter;
        fn num_steps(&self) -> usize;
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
        fn num_steps(&self) -> usize {
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
        fn num_steps(&self) -> usize {
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
        assert_eq!(coords.len(), traverse.num_steps());
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
        assert_eq!(coords.len(), traverse.num_steps().max(2) - 2);
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
        assert_eq!(coords.len(), traverse.num_steps().max(1) - 1);
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
        assert_eq!(coords.len(), traverse.num_steps().max(1) - 1);
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
        LineSegment::new(rand_coord(rng), rand_coord(rng))
    }

    #[test]
    fn all() {
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(1, 1)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(1, 0)));
        test_properties(LineSegment::new(Coord::new(0, 0), Coord::new(2, 1)));
        test_properties(LineSegment::new(Coord::new(1, -1), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(1, 100), Coord::new(0, 0)));
        test_properties(LineSegment::new(Coord::new(100, 1), Coord::new(0, 0)));
        const NUM_RAND_TESTS: usize = 10000;
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
    }
}
