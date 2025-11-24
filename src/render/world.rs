use bevy_ecs::schedule::{ScheduleLabel, Schedules};
use bevy_ecs::world::World;

#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Extract;

struct RenderWorld(World);

fn extract_voxel_objects() {}

impl RenderWorld {
    pub fn new() -> Self {
        let mut world = World::new();

        world
            .get_resource_or_init::<Schedules>()
            .add_systems(Extract, extract_voxel_objects);

        Self(world)
    }

    pub fn extract(&mut self, main_world: &World) {
        todo!()
        // main_world.try_query();
    }
}
