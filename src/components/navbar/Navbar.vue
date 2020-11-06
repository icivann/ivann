<template>
  <div class="left d-flex flex-column h-100 text-center">
    <!-- Build -->
    <div
      class="build tab-button"
      :class="isSelected(editorType.OVERVIEW)"
      @click="switchEditor({editorType: editorType.OVERVIEW})"
    >
      <i class="fas fa-hammer tab-icon"/>
    </div>

    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Model -->
    <div
      class="model tab-button"
      :class="isSelected(editorType.MODEL)"
      @mouseover="displayNavbarContextualMenu(editorType.MODEL)"
      @mouseleave="hideNavbarContextualMenu()"
    >
      <img class="navbar-logo tab-icon" src="@/assets/images/nn_logo.png" alt="IVANN"/>
      <NavbarContextualMenu
        class="navbar-contextual-menu"
        v-if="isModelContextualMenuOpen"
        :editors="modelEditors"
        :editor-type="editorType.MODEL"
      />
    </div>

    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Data -->
    <div
      class="data tab-button"
      :class="isSelected(editorType.DATA)"
      @mouseover="displayNavbarContextualMenu(editorType.DATA)"
      @mouseleave="hideNavbarContextualMenu()"
    >
      <i class="fas fa-database tab-icon"/>
      <NavbarContextualMenu
        class="navbar-contextual-menu"
        v-if="isDataContextualMenuOpen"
        :editors="dataEditors"
        :editor-type="editorType.DATA"
      />
    </div>

    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Train -->
    <div
      class="train tab-button"
      :class="isSelected(editorType.TRAIN)"
      @mouseover="displayNavbarContextualMenu(editorType.TRAIN)"
      @mouseleave="hideNavbarContextualMenu()"
    >
      <i class="fas fa-cogs tab-icon"/>
      <NavbarContextualMenu
        class="navbar-contextual-menu"
        v-if="isTrainContextualMenuOpen"
        :editors="trainEditors"
        :editor-type="editorType.TRAIN"
      />
    </div>

    <!-- Code Vault -->
    <div class="flex-grow-1"/>
    <div
      class="build tab-button"
      :class="inCodeVault && 'selected'"
      @click="enterCodeVault"
    >
      <i class="fab fa-python tab-icon"/>
    </div>
  </div>
</template>

<script lang="ts">
import { Vue, Component } from 'vue-property-decorator';
import EditorType from '@/EditorType';
import { mapGetters, mapMutations } from 'vuex';
import { Getter } from 'vuex-class';
import NavbarContextualMenu from '@/components/navbar/NavbarContextualMenu.vue';

@Component({
  components: { NavbarContextualMenu },
  computed: mapGetters([
    'modelEditors',
    'dataEditors',
    'trainEditors',
    'inCodeVault',
  ]),
  methods: mapMutations(['switchEditor', 'enterCodeVault']),
})
export default class Navbar extends Vue {
  private editorType = EditorType;
  private isModelContextualMenuOpen = false;
  private isDataContextualMenuOpen = false;
  private isTrainContextualMenuOpen = false;
  @Getter('currEditorType') currEditorType!: EditorType;
  @Getter('inCodeVault') inCodeVault!: boolean;

  private isSelected(editorType?: EditorType) {
    return !this.inCodeVault && (this.currEditorType === editorType) ? 'selected' : '';
  }

  private displayNavbarContextualMenu(editorType: EditorType) {
    switch (editorType) {
      case EditorType.MODEL:
        this.isModelContextualMenuOpen = true;
        break;
      case EditorType.DATA:
        this.isDataContextualMenuOpen = true;
        break;
      case EditorType.TRAIN:
        this.isTrainContextualMenuOpen = true;
        break;
      default:
        break;
    }
  }

  private hideNavbarContextualMenu(): void {
    this.isModelContextualMenuOpen = false;
    this.isDataContextualMenuOpen = false;
    this.isTrainContextualMenuOpen = false;
  }
}

</script>

<style scoped>
  .left {
    background: var(--background);
    color: var(--foreground);
    border-right: 0.08rem solid var(--grey);
  }

  .tab-button {
    padding: 0.75rem 0;
    transition-duration: 0.1s;
    border-left-style: solid;
    border-left-width: 1px;
    border-left-color: var(--background);
  }

  .tab-button:hover {
    background: #1c1c1c;
    transition-duration: 0.1s;
    border-left-color: var(--blue);
    cursor: pointer;
  }

  .tab-button.selected {
    border-left-width: 4px;
    border-left-color: var(--blue);
  }

  .tab-icon {
    font-size: 1.5rem;
    height: 1.5rem;
  }

  hr {
    border-top: 0.1rem solid var(--dark-grey) !important;
  }

  .build {
    margin-top: 1rem;
  }

  .navbar-contextual-menu {
    position: absolute;
    margin-top: -35px;
    left: 59px;
    background: var(--background);
    color: var(--foreground);
    z-index: 1;
  }
</style>
