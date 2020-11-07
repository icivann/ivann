<template>
  <div class="h-100">
    <div id="ace">
    </div>
    <div class="save-banner row">
      <div class="col-1 offset-10 my-auto">
        <button class="btn btn-sm btn-secondary pull-right" @click="leaveCodeVault">Cancel</button>
      </div>
      <div class="col-1 my-auto">
        <button class="btn btn-sm btn-primary pull-right" @click="save">Save</button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import Ace from 'brace';
import 'brace/mode/python';
import 'brace/theme/monokai';
import { Getter, Mutation } from 'vuex-class';
import Custom from '@/nodes/model/custom/Custom';

@Component({
  components: {
    Tab,
    Tabs,
  },
})
export default class IdeTab extends Vue {
  @Getter('nodeTriggeringCodeVault') nodeTriggeringCodeVault?: Custom;
  @Getter('inCodeVault') inCodeVault!: boolean;
  @Mutation('leaveCodeVault') leaveCodeVault!: () => void;
  private editor?: Ace.Editor;

  /**
   * Watches the rendering of the CodeVault in order to update code.
   * If the CodeVault was entered from a Custom Node, update code.
   * Else, set empty the editor.
   */
  @Watch('inCodeVault')
  onInCodeVaultChanged(inCodeVault: boolean) {
    if (inCodeVault && this.editor) {
      if (this.nodeTriggeringCodeVault) {
        this.editor.setValue(this.nodeTriggeringCodeVault.getInlineCode());
      } else {
        this.editor.setValue('');
      }
      this.editor.clearSelection();
    }
  }

  mounted() {
    this.editor = Ace.edit('ace');
    this.editor.getSession().setMode('ace/mode/python');
    this.editor.setTheme('ace/theme/monokai');
    this.editor.resize(true);
    if (this.nodeTriggeringCodeVault) {
      this.editor.setValue(this.nodeTriggeringCodeVault.getInlineCode());
      this.editor.clearSelection();
    }
  }

  save() {
    if (this.nodeTriggeringCodeVault && this.editor) {
      this.nodeTriggeringCodeVault.setInlineCode(this.editor.getValue());
    }
    this.leaveCodeVault();
  }
}
</script>

<style scoped>
.save-banner {
  border-top: 0.1rem solid var(--dark-grey);
  background: var(--background);
  height: 3em;
}
#ace {
  height: calc(100% - 3em - 2em);
}
</style>
