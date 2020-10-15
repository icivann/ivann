<template>
  <div class="left-sidebar d-flex flex-column h-100 text-center">
    <div class="model pt-4 pb-3">
      <img class="navbar-logo icon" src="@/assets/images/nn_logo.png" alt="IVANN" />
    </div>
    <div class="data pt-3 pb-3">
      <i class="fas fa-database icon"></i>
    </div>
    <div class="train pt-3">
      <i class="fas fa-cogs icon"></i>
    </div>
    <div class="flex-grow-1" />
    <div class="build pb-4">
      <i class="fas fa-hammer icon"></i>
    </div>
  </div>
</template>

<script>

</script>

<style scoped>
.left-sidebar {
  background: var(--background);
  color: var(--foreground);
  border-right: 0.125vw solid var(--foreground);
}

.icon {
  font-size: 1.5rem;
  height: 1.5rem;
}
</style>
