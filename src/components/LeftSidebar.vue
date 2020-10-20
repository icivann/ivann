<template>
  <div class="left-sidebar d-flex flex-column h-100 text-center">
    <!-- Build -->
    <div class="build pt-4 pb-2">
      <i class="fas fa-hammer tab-icon"></i>
    </div>

    <div class="py-1 px-2"><hr /></div>

    <!-- Model -->
    <div class="model py-2">
      <img class="navbar-logo tab-icon" src="@/assets/images/nn_logo.png" alt="IVANN" />
    </div>
    <div class="py-1 px-2"><hr /></div>

    <!-- Data -->
    <div class="data py-2">
      <i class="fas fa-database tab-icon"></i>
    </div>
    <div class="py-1 px-2"><hr /></div>

    <!-- Train -->
    <div class="train py-2">
      <i class="fas fa-cogs tab-icon"></i>
    </div>
  </div>
</template>

<script>

</script>

<style scoped>
.left-sidebar {
  background: var(--background);
  color: var(--foreground);
  border-right: 0.08rem solid var(--grey);
}

.tab-icon {
  font-size: 1.5rem;
  height: 1.5rem;
}

.add-icon {
  font-size: 1rem;
}

.tab-icon:hover, .add-icon:hover {
  color: var(--blue);
}

hr {
  border-top: 0.1rem solid var(--dark-grey) !important;
}
</style>
